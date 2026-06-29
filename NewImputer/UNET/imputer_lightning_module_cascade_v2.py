import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F


def compute_distance_to_observed(cond_mask):
    """计算每个缺失位置到最近观测位置的距离。
    
    Args:
        cond_mask: [B, K, L]，1表示已观测，0表示缺失
    Returns:
        distance: [B, K, L]，每个位置到最近观测点的距离（已观测点距离为0）
    """
    B, K, L = cond_mask.shape
    device = cond_mask.device
    
    # 用大值初始化距离
    dist = torch.full((B, K, L), fill_value=L, dtype=torch.float, device=device)
    dist[cond_mask == 1] = 0.0
    
    # 前向扫描：从左侧最近观测点的距离
    for t in range(1, L):
        dist[:, :, t] = torch.minimum(dist[:, :, t], dist[:, :, t-1] + 1)
    
    # 后向扫描：取左右两侧观测点距离的较小值
    for t in range(L - 2, -1, -1):
        dist[:, :, t] = torch.minimum(dist[:, :, t], dist[:, :, t+1] + 1)
    
    return dist


def temporal_smoothness_loss(pred, target, eval_mask):
    """时间平滑损失：约束插补结果一阶差分与GT一阶差分一致。
    
    对于缺失位置，要求 pred[t] - pred[t-1] ≈ target[t] - target[t-1]，
    使得插补值在时间维度上保持自然的过渡。
    
    Args:
        pred:      [B, K, L] 模型预测
        target:    [B, K, L] 真实值
        eval_mask: [B, K, L] 评估掩码（缺失位置=1）
    Returns:
        temporal smoothness loss（标量）
    """
    # 计算一阶差分
    pred_diff = pred[:, :, 1:] - pred[:, :, :-1]       # [B, K, L-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]  # [B, K, L-1]
    
    # 差分掩码：只要相邻两个时间步中至少有一个是缺失位置，就纳入计算
    diff_mask = torch.clamp(eval_mask[:, :, 1:] + eval_mask[:, :, :-1], 0, 1)  # [B, K, L-1]
    
    diff_error = torch.abs(pred_diff - target_diff) * diff_mask
    denom = diff_mask.sum()
    
    if denom > 0:
        return diff_error.sum() / denom
    return torch.tensor(0.0, device=pred.device)


def distance_aware_weights(distance, eval_mask, mode="linear", temperature=2.0):
    """根据到最近观测点的距离生成权重。
    
    距离越远，权重越大，迫使模型更关注难以预测的远距离缺失位置。
    
    Args:
        distance:    [B, K, L] 到最近观测点的距离
        eval_mask:   [B, K, L] 评估掩码
        mode:        权重计算模式 "linear" | "sqrt" | "log"
        temperature: 控制权重差异程度
    Returns:
        weights: [B, K, L] 归一化权重（均值为1，仅在eval_mask位置有效）
    """
    if mode == "linear":
        raw_w = 1.0 + distance / temperature
    elif mode == "sqrt":
        raw_w = 1.0 + torch.sqrt(distance) / temperature
    elif mode == "log":
        raw_w = 1.0 + torch.log1p(distance) / temperature
    else:
        raw_w = torch.ones_like(distance)
    
    # 仅在eval_mask位置生效
    raw_w = raw_w * eval_mask
    
    # 归一化使得平均权重为1（保持loss量级不变）
    denom = eval_mask.sum()
    if denom > 0:
        mean_w = raw_w.sum() / denom
        raw_w = raw_w / (mean_w + 1e-8)
    
    return raw_w


class aqi_lightning_module(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        seq_len=36,
        feature_dim=36,
        loss_flag="l1",
        epochs=200,
        decay_flag=True,
        # 低通辅助损失参数
        aux_lowpass_weight=0.1,
        lowpass_kernel_size=5,
        lowpass_mode="avg",
        fft_keep_ratio=0.25,
        # 时间平滑损失参数
        smooth_weight=0.1,
        # 距离感知加权参数
        dist_aware=True,
        dist_mode="sqrt",
        dist_temperature=2.0,
        # 距离感知分层参数（cascade专用）
        dist_layer_aware=True,
    ):
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
        
        # 低通损失
        self.aux_lowpass_weight = aux_lowpass_weight
        self.lowpass_kernel_size = max(1, int(lowpass_kernel_size))
        if self.lowpass_kernel_size % 2 == 0:
            self.lowpass_kernel_size += 1
        self.lowpass_mode = str(lowpass_mode).lower()
        if self.lowpass_mode not in {"avg", "fft"}:
            raise ValueError(f"Unsupported lowpass_mode: {lowpass_mode}. Use 'avg' or 'fft'.")
        self.fft_keep_ratio = float(fft_keep_ratio)
        if not (0.0 < self.fft_keep_ratio <= 1.0):
            raise ValueError("fft_keep_ratio must be in (0, 1].")
        
        # 时间平滑损失
        self.smooth_weight = smooth_weight
        
        # 距离感知加权
        self.dist_aware = dist_aware
        self.dist_mode = dist_mode
        self.dist_temperature = dist_temperature
        self.dist_layer_aware = dist_layer_aware  # 是否对不同层使用不同距离偏好

    def _masked_l1(self, pred, target, mask):
        denom = mask.sum()
        if denom > 0:
            return (torch.abs(target - pred) * mask).sum() / denom
        return torch.tensor(0.0, device=target.device)

    def _weighted_l1(self, pred, target, mask, weights):
        """距离感知加权L1"""
        denom = mask.sum()
        if denom > 0:
            return (torch.abs(target - pred) * mask * weights).sum() / denom
        return torch.tensor(0.0, device=target.device)

    def _temporal_lowpass(self, series):
        if self.lowpass_kernel_size <= 1:
            return series
        bsz, feat_dim, seq_len = series.shape
        pad = self.lowpass_kernel_size // 2
        x = series.reshape(bsz * feat_dim, 1, seq_len)
        x = F.pad(x, (pad, pad), mode="replicate")
        x = F.avg_pool1d(x, kernel_size=self.lowpass_kernel_size, stride=1)
        return x.reshape(bsz, feat_dim, seq_len)

    def _fft_lowpass(self, series):
        _, _, seq_len = series.shape
        spec = torch.fft.rfft(series, dim=-1)
        n_freq = spec.shape[-1]
        keep_bins = max(1, int(n_freq * self.fft_keep_ratio))
        mask = torch.zeros(n_freq, device=series.device, dtype=series.dtype)
        mask[:keep_bins] = 1.0
        spec_filtered = spec * mask.view(1, 1, -1)
        return torch.fft.irfft(spec_filtered, n=seq_len, dim=-1)

    def _build_lowpass_target(self, series):
        if self.lowpass_mode == "avg":
            return self._temporal_lowpass(series)
        return self._fft_lowpass(series)

    def _get_layer_distance_weights(self, distance, eval_mask, layer_idx, num_layers):
        """为不同层生成不同的距离感知权重。
        
        核心思想：前层关注近距离（简单）位置，后层关注远距离（困难）位置。
        与cascade的逐步精化哲学完美契合。
        
        layer 0: 温度大 -> 权重差异小（近远均匀关注，偏简单）
        layer N: 温度小 -> 权重差异大（强调远距离难点）
        """
        if not self.dist_layer_aware:
            return distance_aware_weights(distance, eval_mask, self.dist_mode, self.dist_temperature)
        
        # 前层温度大（权重平坦），后层温度小（权重陡峭）
        layer_ratio = layer_idx / max(num_layers - 1, 1)  # 0 -> 1
        layer_temp = self.dist_temperature * (2.0 - layer_ratio)  # 前层温度高，后层温度低
        
        return distance_aware_weights(distance, eval_mask, self.dist_mode, layer_temp)

    def training_step(self, batch, batch_idx):
        (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
        ) = self.model.process_data(batch)

        x = observed_data * cond_mask
        imputed_results, layer_preds = self.model(coeffs, cond_mask)
        eval_mask = observed_mask - cond_mask

        # 预计算距离（如果启用距离感知）
        dist = None
        if self.dist_aware:
            dist = compute_distance_to_observed(cond_mask)

        # ==================== 主损失 ====================
        if self.loss_flag == "l1":
            if self.dist_aware:
                weights = distance_aware_weights(dist, eval_mask, self.dist_mode, self.dist_temperature)
                base_loss = self._weighted_l1(imputed_results, observed_data, eval_mask, weights)
            else:
                base_loss = self._masked_l1(imputed_results, observed_data, eval_mask)

        elif self.loss_flag == "midEnhance":
            num_layers = len(layer_preds)
            if num_layers > 0:
                layer_weights = torch.linspace(0.2, 1.0, num_layers, dtype=torch.float, device=self.device)
                
                if self.decay_flag:
                    warmup_factor = min(1.0, self.current_epoch / (self.epochs * 0.33))
                else:
                    warmup_factor = 1.0

                weighted_layer_losses = []
                for i, pred in enumerate(layer_preds):
                    abs_error = torch.abs(observed_data - pred)
                    
                    if self.dist_aware:
                        # 每层使用不同的距离权重策略
                        d_weights = self._get_layer_distance_weights(dist, eval_mask, i, num_layers)
                        masked_error = abs_error * eval_mask * d_weights
                    else:
                        masked_error = abs_error * eval_mask
                    
                    current_weight = layer_weights[i] * warmup_factor
                    weighted_layer_losses.append(current_weight * masked_error.sum())

                layer_total = torch.stack(weighted_layer_losses).sum()
                base_loss = layer_total / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
                
                self.log('layer_weighted_loss', base_loss, on_step=False, on_epoch=True)
                self.log('warmup_factor', warmup_factor, on_step=False, on_epoch=True)
            else:
                base_loss = torch.tensor(0.0, device=self.device)

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
                    abs_error = torch.abs(observed_data - pred)
                    
                    if self.dist_aware:
                        d_weights = self._get_layer_distance_weights(dist, eval_mask, i, num_layers)
                        masked_error = abs_error * eval_mask * d_weights
                    else:
                        masked_error = abs_error * eval_mask

                    if i < num_layers - 1:
                        weighted_layer_losses.append(decay_factor * current_weights[i] * masked_error.sum())
                    else:
                        weighted_layer_losses.append(current_weights[i] * masked_error.sum())

                layer_total = torch.stack(weighted_layer_losses).sum()
                base_loss = layer_total / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
            else:
                base_loss = torch.tensor(0.0, device=self.device)
        else:
            raise ValueError(f"Unsupported loss_flag: {self.loss_flag}")

        # ==================== 辅助低通损失 ====================
        aux_lowpass = torch.tensor(0.0, device=self.device)
        if len(layer_preds) > 0 and self.aux_lowpass_weight > 0:
            lowpass_target = self._build_lowpass_target(observed_data)
            aux_lowpass = self._masked_l1(layer_preds[0], lowpass_target, eval_mask)

        # ==================== 时间平滑损失 ====================
        smooth_loss = torch.tensor(0.0, device=self.device)
        if self.smooth_weight > 0:
            smooth_loss = temporal_smoothness_loss(imputed_results, observed_data, eval_mask)

        # ==================== 总损失 ====================
        loss = base_loss + self.aux_lowpass_weight * aux_lowpass + self.smooth_weight * smooth_loss

        # ==================== 日志 ====================
        self.log('base_loss', base_loss, on_step=True, on_epoch=True)
        self.log('aux_lowpass_loss', aux_lowpass, on_step=False, on_epoch=True)
        self.log('smooth_loss', smooth_loss, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        if self.dist_aware and dist is not None:
            # 记录缺失位置平均距离，用于分析
            avg_dist = (dist * eval_mask).sum() / eval_mask.sum() if eval_mask.sum() > 0 else 0
            self.log('avg_missing_dist', avg_dist, on_step=False, on_epoch=True)

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

        mae_current = torch.abs((imputed_results - observed_data) * eval_mask) * scaler
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
