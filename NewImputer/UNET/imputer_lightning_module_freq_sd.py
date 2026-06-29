import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F


def spectral_l1_loss(pred, target, eval_mask):
    """频域辅助损失：对预测和GT分别做FFT，在幅值谱上计算L1。
    
    时域L1约束逐点精度，频域L1约束全局频率分布，
    使得插补结果保持正确的周期性和趋势模式。
    
    Args:
        pred:      [B, K, L] 模型预测
        target:    [B, K, L] 真实值
        eval_mask: [B, K, L] 评估掩码
    Returns:
        频域L1 loss（标量）
    """
    # 构造"完整"信号：已观测位置用GT，缺失位置用模型预测
    completed_pred = target * (1 - eval_mask) + pred * eval_mask   # [B, K, L]
    
    # 沿时间维度做 rFFT
    pred_fft = torch.fft.rfft(completed_pred, dim=-1)    # [B, K, L//2+1] complex
    target_fft = torch.fft.rfft(target, dim=-1)           # [B, K, L//2+1] complex
    
    # 计算幅值谱的 L1 差异
    mag_diff = torch.abs(pred_fft.abs() - target_fft.abs())  # [B, K, L//2+1]
    
    # 只对有缺失值的通道计算频域损失
    channel_has_missing = (eval_mask.sum(dim=-1) > 0).float()  # [B, K]
    channel_has_missing = channel_has_missing.unsqueeze(-1)     # [B, K, 1]
    
    masked_mag_diff = mag_diff * channel_has_missing
    
    if channel_has_missing.sum() > 0:
        return masked_mag_diff.sum() / (channel_has_missing.sum() * mag_diff.shape[-1])
    else:
        return torch.tensor(0.0, device=pred.device)


class aqi_lightning_module(pl.LightningModule):
    def __init__(self, model, lr=1e-3, seq_len=36, feature_dim=36, 
                 loss_flag="freqAux", epochs=200, decay_flag=True,
                 freq_loss_weight=0.1):
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
        self.decay_flag = decay_flag
        self.freq_loss_weight = freq_loss_weight
        torch.use_deterministic_algorithms(True, warn_only=True)

    def training_step(self, batch, batch_idx):
        (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
        ) = self.model.process_data(batch)
        
        x = observed_data * cond_mask
        imputed_results, layer_preds = self.model(coeffs, cond_mask)
        eval_mask = observed_mask - cond_mask
        
        if self.loss_flag == "l1":
            loss = (torch.abs(observed_data - imputed_results) * eval_mask).sum() / eval_mask.sum() if eval_mask.sum() > 0 else 1

        elif self.loss_flag == "freqAux":
            # ============ 频域辅助损失 ============
            # 时域 L1 loss（主损失）
            time_loss = (torch.abs(observed_data - imputed_results) * eval_mask).sum() / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(1.0, device=self.device)
            # 频域 L1 loss（辅助损失）
            freq_loss = spectral_l1_loss(imputed_results, observed_data, eval_mask)
            # 组合
            loss = time_loss + self.freq_loss_weight * freq_loss
            
            self.log('time_loss', time_loss, on_step=False, on_epoch=True)
            self.log('freq_loss', freq_loss, on_step=False, on_epoch=True)
            self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        
        elif self.loss_flag == "freqAux_midEnhance":
            # ============ 频域损失 + 中间监督 联合 ============
            num_layers = len(layer_preds)
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
            
            layer_total = torch.stack(weighted_layer_losses).sum()
            time_loss = layer_total / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
            
            # 频域损失：只在最终预测上算
            freq_loss = spectral_l1_loss(imputed_results, observed_data, eval_mask)
            
            loss = time_loss + self.freq_loss_weight * freq_loss
            
            self.log('time_loss', time_loss, on_step=False, on_epoch=True)
            self.log('freq_loss', freq_loss, on_step=False, on_epoch=True)
            self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
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
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
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
