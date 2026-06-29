"""
v8 Lightning Module: 基于 v5，新增频域损失正则化
L_total = L_main + α·L_curriculum + w_smooth·L_smooth + w_freq·L_freq

L_freq = ||FFT(pred·mask) - FFT(target·mask)||₁ / N_freq
只对缺失位置的预测施加频谱一致性约束
"""
import pytorch_lightning as pl
import torch
import pickle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


def temporal_smoothness_loss(pred, target, eval_mask):
    pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]
    diff_mask = torch.clamp(eval_mask[:, :, 1:] + eval_mask[:, :, :-1], 0, 1)
    diff_error = torch.abs(pred_diff - target_diff) * diff_mask
    denom = diff_mask.sum()
    if denom > 0:
        return diff_error.sum() / denom
    return torch.tensor(0.0, device=pred.device)


def spectral_loss(pred, target, eval_mask):
    """
    频域损失：比较重建序列与真实序列的频谱差异。
    将预测值填入缺失位置，已观测位置保持真实值，
    然后比较整条序列的频谱。
    
    pred, target, eval_mask: (B, K, L)
    """
    # 构造完整序列：观测位置用真实值，缺失位置用预测值
    full_pred = target * (1 - eval_mask) + pred * eval_mask
    
    # 对时间维度做 FFT
    pred_fft = torch.fft.rfft(full_pred, dim=-1)  # (B, K, L//2+1) complex
    target_fft = torch.fft.rfft(target, dim=-1)
    
    # 幅度谱的 L1 差异
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    loss = torch.abs(pred_mag - target_mag).mean()
    return loss


class aqi_lightning_module(pl.LightningModule):
    def __init__(self, model, lr=1e-3, seq_len=36, feature_dim=36,
                 epochs=200, smooth_weight=0.0,
                 curriculum_weight=0.3,
                 curriculum_warmup_ratio=0.1,
                 curriculum_temperature=1.0,
                 freq_loss_weight=0.0,
                 ):
        super().__init__()
        path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.model = model
        self.lr = lr
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.epochs = epochs
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.smooth_weight = smooth_weight
        self.curriculum_weight = curriculum_weight
        self.curriculum_warmup_ratio = curriculum_warmup_ratio
        self.curriculum_temperature = curriculum_temperature
        self.freq_loss_weight = freq_loss_weight

    def _masked_l1(self, pred, target, mask):
        denom = mask.sum()
        if denom > 0:
            return (torch.abs(target - pred) * mask).sum() / denom
        return torch.tensor(0.0, device=target.device)

    def _weighted_masked_l1(self, pred, target, mask, weights):
        w_mask = mask * weights
        denom = w_mask.sum()
        if denom > 0:
            return (torch.abs(target - pred) * w_mask).sum() / denom
        return torch.tensor(0.0, device=target.device)

    def _build_soft_difficulty_weights(self, layer_preds, observed_data, eval_mask):
        num_layers = len(layer_preds)
        with torch.no_grad():
            error_map = torch.abs(layer_preds[-1].detach() - observed_data) * eval_mask
        valid_errors = error_map[eval_mask.bool()]
        if valid_errors.numel() == 0:
            return [torch.ones_like(eval_mask)] * (num_layers - 1)
        e_min = valid_errors.min()
        e_max = valid_errors.max()
        if e_max - e_min < 1e-8:
            return [torch.ones_like(eval_mask)] * (num_layers - 1)
        norm_error = (error_map - e_min) / (e_max - e_min + 1e-8)
        weights_list = []
        for l in range(num_layers - 1):
            ease_factor = 1.0 - l / max(num_layers - 1, 1)
            log_weights = -ease_factor * norm_error / self.curriculum_temperature
            log_weights = log_weights * eval_mask
            weights = torch.exp(log_weights) * eval_mask
            w_sum = (weights * eval_mask).sum()
            n_eval = eval_mask.sum()
            if w_sum > 0 and n_eval > 0:
                weights = weights * (n_eval / w_sum)
            weights_list.append(weights)
        return weights_list

    def _get_curriculum_alpha(self):
        warmup_end = self.epochs * self.curriculum_warmup_ratio
        if warmup_end > 0 and self.current_epoch < warmup_end:
            return 0.0
        return self.curriculum_weight

    def training_step(self, batch, batch_idx):
        (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
        ) = self.model.process_data(batch)

        imputed_results, layer_preds = self.model(coeffs, cond_mask)
        eval_mask = observed_mask - cond_mask
        num_layers = len(layer_preds)

        # L_main
        L_main = self._masked_l1(layer_preds[-1], observed_data, eval_mask)

        # L_curriculum
        alpha = self._get_curriculum_alpha()
        L_curriculum = torch.tensor(0.0, device=self.device)
        if alpha > 0 and num_layers > 1:
            soft_weights = self._build_soft_difficulty_weights(layer_preds, observed_data, eval_mask)
            curr_losses = []
            for l in range(num_layers - 1):
                layer_loss = self._weighted_masked_l1(
                    layer_preds[l], observed_data, eval_mask, soft_weights[l]
                )
                curr_losses.append(layer_loss)
            L_curriculum = torch.stack(curr_losses).mean()

        # L_smooth
        smooth_loss = torch.tensor(0.0, device=self.device)
        if self.smooth_weight > 0:
            smooth_loss = temporal_smoothness_loss(imputed_results, observed_data, eval_mask)

        # L_freq: 频域损失正则化
        freq_loss = torch.tensor(0.0, device=self.device)
        if self.freq_loss_weight > 0:
            freq_loss = spectral_loss(imputed_results, observed_data, eval_mask)

        # 总损失
        loss = L_main + alpha * L_curriculum + self.smooth_weight * smooth_loss + self.freq_loss_weight * freq_loss

        self.log('L_main', L_main, on_step=False, on_epoch=True)
        self.log('L_curriculum', L_curriculum, on_step=False, on_epoch=True)
        self.log('smooth_loss', smooth_loss, on_step=False, on_epoch=True)
        self.log('freq_loss', freq_loss, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)

        for i in range(num_layers):
            layer_mae = self._masked_l1(layer_preds[i], observed_data, eval_mask)
            self.log(f'layer_{i}_mae', layer_mae, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
        ) = self.model.process_data(batch)

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
