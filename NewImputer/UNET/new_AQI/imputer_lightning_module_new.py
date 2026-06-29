import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F



class aqi_lightning_module(pl.LightningModule):
    def __init__(self, model, lr=1e-3, seq_len=36, feature_dim=36, loss_flag="l1",epochs=200, decay_flag=True):
        super().__init__()
        path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.model = model
        self.lr = lr
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.loss_flag = loss_flag  # "l1" or "threshold" or "midSupervision"
        self.epochs = epochs
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.decay_flag = decay_flag

    def training_step(self, batch, batch_idx):
        # Extract data from batch using the model's process_data method
        (
            observed_data,    # [B, K, L]
            observed_mask,    # [B, K, L]
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,        # [B, K, L]
        ) = self.model.process_data(batch)
        layer_mask_1 = batch["layer_mask_1"].to(self.device).float().permute(0, 2, 1)
        layer_mask_2 = batch["layer_mask_2"].to(self.device).float().permute(0, 2, 1)
        layer_mask_3 = batch["layer_mask_3"].to(self.device).float().permute(0, 2, 1)
        
        # Prepare input: mask observed data with conditional mask
        x = observed_data * cond_mask  # [B, K, L]

        #imputed_results, layer_preds = self.model(freq_coeffs, cond_mask)   # [B, K, L]
        imputed_results, layer_preds = self.model(coeffs, cond_mask)   # [B, K, L]
        
        # Compute evaluation mask (positions to evaluate imputation)
        eval_mask = observed_mask - cond_mask  # [B, K, L]

        # Calculate MSE loss only on imputed positions
        if self.loss_flag == "l1":
            loss = (torch.abs(observed_data - imputed_results) * eval_mask).sum() / eval_mask.sum() if eval_mask.sum() > 0 else 1

        elif self.loss_flag == "threshold":       
            theta = 0.5  # Example threshold value, can be adjusted for data
            alpha = 0.5  # Example amplification factor to enlarge errors above the threshold

            # Calculate absolute error
            abs_errors = torch.abs(observed_data - imputed_results) * eval_mask

            # Create threshold mask
            threshold_mask = (abs_errors > theta).float()

            # Adjust error by applying amplification factor for thresholded values
            adjusted_errors = abs_errors * (1 + (alpha - 1) * threshold_mask)

            # Compute the loss
            loss = adjusted_errors.sum() / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(1.0, device=observed_data.device)
        
        elif self.loss_flag == "midSupervision":
            final_loss = self.masked_l1(imputed_results, observed_data, eval_mask)

            # 分层辅助监督
            loss_l1 = self.masked_l1(layer_preds[0], observed_data, layer_mask_1)
            loss_l2 = self.masked_l1(layer_preds[1], observed_data, layer_mask_2)

            # 逐层变好约束：更深层不能比更浅层差
            prog12 = self.progressive_better_loss(
                layer_preds[0], layer_preds[1], observed_data, layer_mask_1, margin=0.0
            )
            prog23 = self.progressive_better_loss(
                layer_preds[1], layer_preds[2], observed_data, layer_mask_2, margin=0.0
            )

            loss = (
                1.0 * final_loss
                + 0.15 * loss_l1
                + 0.25 * loss_l2
                # + 0.10 * prog12
                # + 0.10 * prog23 
            )

            self.log("final_loss", final_loss, on_step=False, on_epoch=True)
            self.log("loss_l1_easy", loss_l1, on_step=False, on_epoch=True)
            self.log("loss_l2_easymed", loss_l2, on_step=False, on_epoch=True)
            self.log("prog12", prog12, on_step=False, on_epoch=True)
            self.log("prog23", prog23, on_step=False, on_epoch=True)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Same data extraction as in training_step
        (
            observed_data,    # [B, K, L]
            observed_mask,    # [B, K, L]
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,        # [B, K, L]
        ) = self.model.process_data(batch)

        # Prepare input
        x = observed_data * cond_mask  # [B, K, L]
        
        # Forward pass
        #imputed_results = self.model(x,cond_mask)   # [B, K, L]
        #imputed_results = self.model(coeffs,cond_mask)  # [B, K, L]
        
        #imputed_results,_ = self.model(freq_coeffs,cond_mask,timeofday,dayofweek)   # [B, K, L]
        imputed_results,_ = self.model(coeffs,cond_mask)   # [B, K, L]
        # Compute evaluation mask
        eval_mask = observed_mask - cond_mask  # [B, K, L]

        # Calculate validation loss
        loss = ((observed_data - imputed_results) ** 2 * eval_mask).sum() / eval_mask.sum()
        
        # Log validation loss
        val_mae = (torch.abs(observed_data - imputed_results) * eval_mask).sum() / eval_mask.sum()
        val_mse = ((observed_data - imputed_results) ** 2 * eval_mask).sum() / eval_mask.sum()

        self.log('val_mae', val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_mse, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_start(self):
        self.total_mae = 0
        self.total_mse = 0
        self.evalpoints_total = 0
    
    def test_step(self, batch, batch_idx):
        scaler = torch.from_numpy(self.train_std).to(self.device).float()
        mean_scaler = torch.from_numpy(self.train_mean).to(self.device).float()
        (
            observed_data,    # [B, K, L]
            observed_mask,    # [B, K, L]
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,        # [B, K, L]
        ) = self.model.process_data(batch)

        x = observed_data * cond_mask
        #imputed_results = self.model(x,cond_mask)
        #imputed_results = self.model(coeffs,cond_mask)  # [B, K, L]
        #imputed_results,_ = self.model(freq_coeffs,cond_mask,timeofday,dayofweek)  # [B, K, L]
        imputed_results,_ = self.model(coeffs,cond_mask)  # [B, K, L]

        #############
        imputed_results = imputed_results.permute(0,2,1)  # [B, L, K] 
        eval_mask = observed_mask - cond_mask
        # for i in range(len(cut_length)):  # to avoid double evaluation
        #             eval_mask[i, ..., 0 : cut_length[i].item()] = 0
        eval_mask = eval_mask.permute(0,2,1)  # [B, L, K]
        observed_data = observed_data.permute(0,2,1)  # [B, L, K]
        #############


        mae_current = (
                torch.abs((imputed_results-observed_data)*eval_mask)
            )*scaler
        mse_current = (
            ((imputed_results-observed_data)*eval_mask)**2
        )*(scaler**2)
        self.total_mae += mae_current.sum().item()
        self.total_mse += mse_current.sum().item()
        self.evalpoints_total += eval_mask.sum().item()
    
    def on_test_epoch_end(self):
        print(self.evalpoints_total)
        MAE = self.total_mae / self.evalpoints_total
        MSE = self.total_mse / self.evalpoints_total
        
        # logs
        self.log('test_MAE', MAE)
        self.log('test_MSE', MSE)

        print(f"Test Set Average MAE: {MAE}")
        print(f"Test Set Average MSE: {MSE}")
        
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    #     # 预热阶段：5 个 epoch 线性增加
    #     warmup_epochs = 5
    #     warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0))
        
    #     # 余弦退火
    #     cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
    #     )
        
    #     # 组合调度器
    #     scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])
        
    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': {
    #             'scheduler': scheduler,
    #             'interval': 'epoch'
    #         }
    #     }
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Warmup stage: linearly increases over 5 epochs
        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0))
        
        # Cosine Annealing
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
        )
        
        # Sequential scheduler: first warmup for `warmup_epochs` epochs, then switch to cosine scheduler
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def masked_l1(self, pred, target, mask):
        denom = mask.sum().clamp_min(1.0)
        return (torch.abs(pred - target) * mask).sum() / denom


    def progressive_better_loss(self, shallow_pred, deep_pred, target, mask, margin=0.0):
        """
        要求 deep_pred 在给定 mask 上不要比 shallow_pred 更差
        """
        denom = mask.sum().clamp_min(1.0)
        err_shallow = torch.abs(shallow_pred - target) * mask
        err_deep = torch.abs(deep_pred - target) * mask
        return F.relu(err_deep - err_shallow + margin).sum() / denom