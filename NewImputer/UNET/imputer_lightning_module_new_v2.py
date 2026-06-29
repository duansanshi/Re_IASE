import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F


def temporal_smoothness_loss(pred, target, eval_mask):
    """时间平滑损失：约束插补结果一阶差分与GT一阶差分一致。
    
    Args:
        pred:      [B, K, L] 模型预测
        target:    [B, K, L] 真实值
        eval_mask: [B, K, L] 评估掩码（缺失位置=1）
    Returns:
        temporal smoothness loss（标量）
    """
    pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]
    
    # 相邻两个时间步中至少有一个是缺失位置，就纳入计算
    diff_mask = torch.clamp(eval_mask[:, :, 1:] + eval_mask[:, :, :-1], 0, 1)
    
    diff_error = torch.abs(pred_diff - target_diff) * diff_mask
    denom = diff_mask.sum()
    
    if denom > 0:
        return diff_error.sum() / denom
    return torch.tensor(0.0, device=pred.device)


class aqi_lightning_module(pl.LightningModule):
    def __init__(self, model, lr=1e-3, seq_len=36, feature_dim=36, loss_flag="l1",
                 epochs=200, decay_flag=True, smooth_weight=0.0):
        super().__init__()
        path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.model = model
        self.lr = lr
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.loss_flag = loss_flag
        self.epochs = epochs
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.decay_flag = decay_flag
        self.smooth_weight = smooth_weight

    def training_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        ) = self.model.process_data(batch)
        
        x = observed_data * cond_mask
        imputed_results, layer_preds = self.model(coeffs, cond_mask)
        eval_mask = observed_mask - cond_mask

        # ==================== 主损失 ====================
        if self.loss_flag == "l1":
            base_loss = (torch.abs(observed_data - imputed_results) * eval_mask).sum() / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(1.0, device=self.device)

        elif self.loss_flag == "midSupervision":
            num_layers = len(layer_preds)
            if num_layers > 0:
                max_weight = 0.1
                gamma = 2.0
                base_weights = torch.tensor([gamma**(i + 1) for i in range(num_layers)], dtype=torch.float, device=self.device)
                scale_factor = max_weight / base_weights[-1]
                current_weights = base_weights * scale_factor

                if self.decay_flag:
                    decay_factor = max(0, 1.0 - self.current_epoch / (self.epochs * 0.5))
                else:
                    decay_factor = 1.0

                weighted_layer_losses = []
                for i, pred in enumerate(layer_preds):
                    current_weight = current_weights[i]
                    abs_error = torch.abs(observed_data - pred)
                    masked_error = abs_error * eval_mask
                    if i < num_layers - 1:
                        weighted_layer_losses.append(decay_factor * current_weight * masked_error.sum())
                    else:
                        weighted_layer_losses.append(current_weight * masked_error.sum())

                layer_total_weighted_sum = torch.stack(weighted_layer_losses).sum()
                base_loss = layer_total_weighted_sum / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
                self.log('layer_weighted_loss', base_loss, on_step=False, on_epoch=True)
            else:
                base_loss = torch.tensor(0.0, device=self.device)
        
        elif self.loss_flag == "midEnhance":
            num_layers = len(layer_preds)
            if num_layers > 0:
                weights = torch.linspace(0.2, 1.0, num_layers, dtype=torch.float, device=self.device)
                if self.decay_flag:
                    warmup_factor = min(1.0, self.current_epoch / (self.epochs * 0.33))
                else:
                    warmup_factor = 1.0

                weighted_layer_losses = []
                for i, pred in enumerate(layer_preds):
                    abs_error = torch.abs(observed_data - pred)
                    masked_error = abs_error * eval_mask
                    current_weight = weights[i] * warmup_factor
                    weighted_layer_losses.append(current_weight * masked_error.sum())

                layer_total_weighted_sum = torch.stack(weighted_layer_losses).sum()
                base_loss = layer_total_weighted_sum / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
                
                self.log('layer_weighted_loss', base_loss, on_step=False, on_epoch=True)
                self.log('warmup_factor', warmup_factor, on_step=False, on_epoch=True)
                for i, w in enumerate(weights):
                    self.log(f'layer_weight_{i}', w * warmup_factor, on_step=False, on_epoch=True)
            else:
                base_loss = torch.tensor(0.0, device=self.device)
        else:
            raise ValueError(f"Unsupported loss_flag: {self.loss_flag}")

        # ==================== 时间平滑损失 ====================
        smooth_loss = torch.tensor(0.0, device=self.device)
        if self.smooth_weight > 0:
            smooth_loss = temporal_smoothness_loss(imputed_results, observed_data, eval_mask)

        # ==================== 总损失 ====================
        loss = base_loss + self.smooth_weight * smooth_loss

        self.log('base_loss', base_loss, on_step=False, on_epoch=True)
        self.log('smooth_loss', smooth_loss, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        ) = self.model.process_data(batch)

        x = observed_data * cond_mask
        imputed_results, _ = self.model(coeffs, cond_mask)
        eval_mask = observed_mask - cond_mask
        loss = ((observed_data - imputed_results) ** 2 * eval_mask).sum() / eval_mask.sum()
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
        ) = self.model.process_data(batch)

        imputed_results, _ = self.model(coeffs, cond_mask)

        imputed_results = imputed_results.permute(0, 2, 1)
        eval_mask = (observed_mask - cond_mask).permute(0, 2, 1)
        observed_data = observed_data.permute(0, 2, 1)

        mae_current = (torch.abs((imputed_results - observed_data) * eval_mask)) * scaler
        mse_current = (((imputed_results - observed_data) * eval_mask) ** 2) * (scaler ** 2)
        self.total_mae += mae_current.sum().item()
        self.total_mse += mse_current.sum().item()
        self.evalpoints_total += eval_mask.sum().item()
    
    def on_test_epoch_end(self):
        print(self.evalpoints_total)
        MAE = self.total_mae / self.evalpoints_total
        MSE = self.total_mse / self.evalpoints_total
        self.log('test_MAE', MAE)
        self.log('test_MSE', MSE)
        print(f"Test Set Average MAE: {MAE}")
        print(f"Test Set Average MSE: {MSE}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0))
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.lr * 0.1)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }
