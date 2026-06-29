import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import LambdaLR, ChainedScheduler


class aqi_lightning_module(pl.LightningModule):
    def __init__(self, model, lr=1e-3, seq_len=36, feature_dim=36):
        super().__init__()
        path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.model = model
        self.lr = lr
        self.seq_len = seq_len
        self.feature_dim = feature_dim

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
            freq_coeffs,
            timeofday,
            dayofweek
        ) = self.model.process_data(batch)
        
        # Prepare input: mask observed data with conditional mask
        x = observed_data * cond_mask  # [B, K, L]
        
        # Forward pass through the model
        #imputed_results = self.model(x,cond_mask)  # [B, K, L]
        #imputed_results = self.model(coeffs,cond_mask)  # [B, K, L]
        imputed_results = self.model(freq_coeffs,cond_mask,timeofday,dayofweek)  # [B, K, L]

        # Compute evaluation mask (positions to evaluate imputation)
        eval_mask = observed_mask - cond_mask  # [B, K, L]

        # Calculate MSE loss only on imputed positions
        # loss = ((observed_data - imputed_results) ** 2 * eval_mask).sum() / eval_mask.sum() if eval_mask.sum()>0 else 1
        # loss = (torch.abs(observed_data - imputed_results) * eval_mask).sum() / eval_mask.sum() if eval_mask.sum()>0 else 1

        #定义阈值和放大因子
        theta = 0.5  # 示例阈值，可根据数据调整
        alpha = 0.5  # 示例放大因子，对超过阈值的误差放大两倍

        # 计算绝对误差
        #abs_errors = torch.abs(observed_data - imputed_results)**2 * eval_mask
        abs_errors = torch.abs(observed_data - imputed_results)* eval_mask

        # 创建超过阈值的掩码
        threshold_mask = (abs_errors > theta).float()

        # 调整误差：对超过阈值的部分应用放大因子
        adjusted_errors = abs_errors * (1 + (alpha - 1) * threshold_mask)

        # 计算损失
        loss = adjusted_errors.sum() / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(1.0, device=observed_data.device)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Same data extraction as in training_step
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            freq_coeffs,
            timeofday,
            dayofweek
        ) = self.model.process_data(batch)

        # Prepare input
        x = observed_data * cond_mask  # [B, K, L]
        
        # Forward pass
        #imputed_results = self.model(x,cond_mask)   # [B, K, L]
        #imputed_results = self.model(coeffs,cond_mask)  # [B, K, L]
        
        imputed_results = self.model(freq_coeffs,cond_mask,timeofday,dayofweek)   # [B, K, L]
        # Compute evaluation mask
        eval_mask = observed_mask - cond_mask  # [B, K, L]

        # Calculate validation loss
        loss = ((observed_data - imputed_results) ** 2 * eval_mask).sum() / eval_mask.sum()
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_start(self):
        self.total_mae = 0
        self.total_mse = 0
        self.evalpoints_total = 0
    
    def test_step(self, batch, batch_idx):
        scaler = torch.from_numpy(self.train_std).to(self.device).float()
        mean_scaler = torch.from_numpy(self.train_mean).to(self.device).float()
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            freq_coeffs,
            timeofday,
            dayofweek
        ) = self.model.process_data(batch)

        x = observed_data * cond_mask
        #imputed_results = self.model(x,cond_mask)
        #imputed_results = self.model(coeffs,cond_mask)  # [B, K, L]
        imputed_results = self.model(freq_coeffs,cond_mask,timeofday,dayofweek)  # [B, K, L]
        eval_mask = observed_mask - cond_mask
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
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 预热阶段：5 个 epoch 线性增加
        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0))
        
        # 余弦退火
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=self.lr * 0.1
        )
        
        # 组合调度器
        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }