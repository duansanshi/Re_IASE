import pytorch_lightning as pl
import torch
import torch.nn as nn
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


class pems04_lightning_module(pl.LightningModule):
    def __init__(self, model, lr=1e-3, seq_len=24, feature_dim=307,
                 epochs=40, smooth_weight=0.0,
                 curriculum_weight=0.3,
                 curriculum_warmup_ratio=0.1,
                 ):
        super().__init__()
        path = "/home/duanlei/PriSTI/data/PEMS04/pems04_meanstd.pk"
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

    def _masked_l1(self, pred, target, mask):
        denom = mask.sum()
        if denom > 0:
            return (torch.abs(target - pred) * mask).sum() / denom
        return torch.tensor(0.0, device=target.device)

    def _build_difficulty_masks(self, layer_preds, observed_data, eval_mask):
        num_layers = len(layer_preds)
        with torch.no_grad():
            error_map = torch.abs(layer_preds[-1].detach() - observed_data) * eval_mask

        valid_errors = error_map[eval_mask.bool()]
        if valid_errors.numel() == 0:
            return [eval_mask] * num_layers

        masks = []
        for i in range(num_layers):
            if i == num_layers - 1:
                masks.append(eval_mask)
            else:
                q = (i + 1) / num_layers
                threshold = torch.quantile(valid_errors, q)
                difficulty_mask = ((error_map <= threshold) & eval_mask.bool()).float()
                if difficulty_mask.sum() < 1:
                    difficulty_mask = eval_mask
                masks.append(difficulty_mask)
        return masks

    def _get_curriculum_alpha(self):
        warmup_end = self.epochs * self.curriculum_warmup_ratio
        if warmup_end > 0 and self.current_epoch < warmup_end:
            return 0.0
        return self.curriculum_weight

    def training_step(self, batch, batch_idx):
        (
            observed_data, observed_mask, observed_tp,
            gt_mask, cut_length,
            coeffs, cond_mask,
        ) = self.model.process_data(batch)

        x = observed_data * cond_mask
        imputed_results, layer_preds = self.model(coeffs, cond_mask)
        eval_mask = observed_mask - cond_mask
        num_layers = len(layer_preds)

        L_main = self._masked_l1(layer_preds[-1], observed_data, eval_mask)

        alpha = self._get_curriculum_alpha()
        L_curriculum = torch.tensor(0.0, device=self.device)

        if alpha > 0 and num_layers > 1:
            difficulty_masks = self._build_difficulty_masks(layer_preds, observed_data, eval_mask)
            curr_losses = []
            for l in range(num_layers - 1):
                layer_loss = self._masked_l1(layer_preds[l], observed_data, difficulty_masks[l])
                curr_losses.append(layer_loss)
            L_curriculum = torch.stack(curr_losses).mean()

        smooth_loss = torch.tensor(0.0, device=self.device)
        if self.smooth_weight > 0:
            smooth_loss = temporal_smoothness_loss(imputed_results, observed_data, eval_mask)

        loss = L_main + alpha * L_curriculum + self.smooth_weight * smooth_loss

        self.log('L_main', L_main, on_step=False, on_epoch=True)
        self.log('L_curriculum', L_curriculum, on_step=False, on_epoch=True)
        self.log('alpha', alpha, on_step=False, on_epoch=True)
        self.log('smooth_loss', smooth_loss, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)

        for i in range(num_layers):
            layer_mae = self._masked_l1(layer_preds[i], observed_data, eval_mask)
            self.log(f'layer_{i}_mae', layer_mae, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (
            observed_data, observed_mask, observed_tp,
            gt_mask, cut_length,
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
            gt_mask, cut_length,
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
