import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F


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
        aux_lowpass_weight=0.1,
        lowpass_kernel_size=5,
        lowpass_mode="avg",
        fft_keep_ratio=0.25,
    ):
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

    def _masked_l1(self, pred, target, mask):
        denom = mask.sum()
        if denom > 0:
            return (torch.abs(target - pred) * mask).sum() / denom
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

        if self.loss_flag == "l1":
            base_loss = self._masked_l1(imputed_results, observed_data, eval_mask)

        elif self.loss_flag == "threshold":
            theta = 0.5
            alpha = 0.5

            abs_errors = torch.abs(observed_data - imputed_results) * eval_mask
            threshold_mask = (abs_errors > theta).float()
            adjusted_errors = abs_errors * (1 + (alpha - 1) * threshold_mask)
            base_loss = adjusted_errors.sum() / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(0.0, device=observed_data.device)

        elif self.loss_flag == "midSupervision":
            num_layers = len(layer_preds)

            if num_layers > 0:
                max_weight = 0.1
                gamma = 2.0

                base_weights = torch.tensor([gamma**(i + 1) for i in range(num_layers)], dtype=torch.float, device=self.device)
                scale_factor = max_weight / base_weights[-1]
                current_weights = base_weights * scale_factor

                weighted_layer_losses = []

                if self.decay_flag:
                    decay_factor = max(0, 1.0 - self.current_epoch / (self.epochs * 0.5))
                else:
                    decay_factor = 1.0

                for i, pred in enumerate(layer_preds):
                    current_weight = current_weights[i]
                    abs_error = torch.abs(observed_data - pred)
                    masked_error = abs_error * eval_mask

                    if i < num_layers - 1:
                        weighted_layer_losses.append(decay_factor * current_weight * masked_error.sum())
                    else:
                        weighted_layer_losses.append(current_weight * masked_error.sum())

                layer_total_weighted_sum = torch.stack(weighted_layer_losses).sum()

                if eval_mask.sum() > 0:
                    layer_weighted_loss = layer_total_weighted_sum / eval_mask.sum()
                else:
                    layer_weighted_loss = torch.tensor(0.0, device=self.device)

                base_loss = layer_weighted_loss
                self.log('layer_weighted_loss', layer_weighted_loss, on_step=False, on_epoch=True)
            else:
                base_loss = torch.tensor(0.0, device=observed_data.device)

        elif self.loss_flag == "midEnhance":
            num_layers = len(layer_preds)

            if num_layers > 0:
                weights = torch.linspace(0.2, 1.0, num_layers, dtype=torch.float, device=self.device)

                if self.decay_flag:
                    warmup_factor = min(1.0, self.current_epoch / (self.epochs * 0.33))
                else:
                    warmup_factor = 1.0

                weighted_layer_losses = []
                total_abs_error = 0

                for i, pred in enumerate(layer_preds):
                    abs_error = torch.abs(observed_data - pred)
                    masked_error = abs_error * eval_mask

                    current_weight = weights[i] * warmup_factor
                    weighted_layer_losses.append(current_weight * masked_error.sum())

                    if i == num_layers - 1:
                        total_abs_error = masked_error.sum()

                layer_total_weighted_sum = torch.stack(weighted_layer_losses).sum()
                if eval_mask.sum() > 0:
                    layer_weighted_loss = layer_total_weighted_sum / eval_mask.sum()
                else:
                    layer_weighted_loss = torch.tensor(0.0, device=self.device)

                base_loss = layer_weighted_loss
                self.log('layer_weighted_loss', layer_weighted_loss, on_step=False, on_epoch=True)
                self.log('warmup_factor', warmup_factor, on_step=False, on_epoch=True)

                for i, w in enumerate(weights):
                    self.log(f'layer_weight_{i}', w * warmup_factor, on_step=False, on_epoch=True)
            else:
                base_loss = torch.tensor(0.0, device=observed_data.device)

        else:
            raise ValueError(f"Unsupported loss_flag: {self.loss_flag}")

        aux_loss = torch.tensor(0.0, device=observed_data.device)
        if len(layer_preds) > 0 and self.aux_lowpass_weight > 0:
            lowpass_target = self._build_lowpass_target(observed_data)
            aux_loss = self._masked_l1(layer_preds[0], lowpass_target, eval_mask)

        loss = base_loss + self.aux_lowpass_weight * aux_loss
        self.log('base_loss', base_loss, on_step=True, on_epoch=True)
        self.log('aux_lowpass_loss', aux_loss, on_step=True, on_epoch=True)
        self.log('aux_lowpass_weight', torch.tensor(self.aux_lowpass_weight, device=observed_data.device), on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

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

        x = observed_data * cond_mask
        imputed_results, _ = self.model(coeffs, cond_mask)

        imputed_results = imputed_results.permute(0, 2, 1)
        eval_mask = observed_mask - cond_mask
        eval_mask = eval_mask.permute(0, 2, 1)
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

        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
        )

        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
