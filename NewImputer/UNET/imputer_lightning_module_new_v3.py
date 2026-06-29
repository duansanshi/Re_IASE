import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F


def temporal_smoothness_loss(pred, target, eval_mask):
    """时间平滑损失"""
    pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]
    diff_mask = torch.clamp(eval_mask[:, :, 1:] + eval_mask[:, :, :-1], 0, 1)
    diff_error = torch.abs(pred_diff - target_diff) * diff_mask
    denom = diff_mask.sum()
    if denom > 0:
        return diff_error.sum() / denom
    return torch.tensor(0.0, device=pred.device)


class aqi_lightning_module(pl.LightningModule):
    def __init__(self, model, lr=1e-3, seq_len=36, feature_dim=36,
                 epochs=200, smooth_weight=0.0,
                 # 层间一致性蒸馏参数
                 prog_max_weight=0.3,      # α(t) 最大值
                 prog_warmup_ratio=0.2,    # 前多少比例的epoch做warmup
                 prog_lambda_mode="equal", # 各层λ_l分配: "equal" | "decay" | "increase"
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
        self.prog_max_weight = prog_max_weight
        self.prog_warmup_ratio = prog_warmup_ratio
        self.prog_lambda_mode = prog_lambda_mode

    def _masked_l1(self, pred, target, mask):
        denom = mask.sum()
        if denom > 0:
            return (torch.abs(target - pred) * mask).sum() / denom
        return torch.tensor(0.0, device=target.device)

    def _get_prog_alpha(self):
        """warm-up系数 α(t): 前 prog_warmup_ratio 的epoch从0线性升到 prog_max_weight"""
        warmup_end = self.epochs * self.prog_warmup_ratio
        if warmup_end > 0:
            alpha = min(self.current_epoch / warmup_end, 1.0) * self.prog_max_weight
        else:
            alpha = self.prog_max_weight
        return alpha

    def _get_layer_lambdas(self, num_inter_layers):
        """各中间层的λ_l权重"""
        if num_inter_layers == 0:
            return []
        if self.prog_lambda_mode == "equal":
            return [1.0] * num_inter_layers
        elif self.prog_lambda_mode == "decay":
            # 越浅的层权重越小
            return [0.5 + 0.5 * i / max(num_inter_layers - 1, 1) for i in range(num_inter_layers)]
        elif self.prog_lambda_mode == "increase":
            # 越浅的层权重越大（鼓励浅层更快对齐）
            return [1.0 - 0.5 * i / max(num_inter_layers - 1, 1) for i in range(num_inter_layers)]
        else:
            return [1.0] * num_inter_layers

    def training_step(self, batch, batch_idx):
        (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
        ) = self.model.process_data(batch)
        
        x = observed_data * cond_mask
        imputed_results, layer_preds = self.model(coeffs, cond_mask)
        eval_mask = observed_mask - cond_mask
        num_layers = len(layer_preds)

        # ==================== L_main: 最终层直接对GT监督 ====================
        L_main = self._masked_l1(layer_preds[-1], observed_data, eval_mask)

        # ==================== L_prog: 层间一致性蒸馏 ====================
        L_prog = torch.tensor(0.0, device=self.device)
        if num_layers > 1:
            lambdas = self._get_layer_lambdas(num_layers - 1)
            prog_losses = []
            for l in range(num_layers - 1):
                # 第l层去模仿第l+1层的预测，对第l+1层做stop-gradient
                teacher = layer_preds[l + 1].detach()  # sg(X^(l+1))
                student = layer_preds[l]               # X^(l)
                layer_loss = self._masked_l1(student, teacher, eval_mask)
                prog_losses.append(lambdas[l] * layer_loss)
            
            L_prog = torch.stack(prog_losses).sum()

        alpha = self._get_prog_alpha()

        # ==================== 时间平滑损失 ====================
        smooth_loss = torch.tensor(0.0, device=self.device)
        if self.smooth_weight > 0:
            smooth_loss = temporal_smoothness_loss(imputed_results, observed_data, eval_mask)

        # ==================== 总损失 ====================
        loss = L_main + alpha * L_prog + self.smooth_weight * smooth_loss

        # ==================== 日志 ====================
        self.log('L_main', L_main, on_step=False, on_epoch=True)
        self.log('L_prog', L_prog, on_step=False, on_epoch=True)
        self.log('alpha', alpha, on_step=False, on_epoch=True)
        self.log('smooth_loss', smooth_loss, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        
        # 记录各层与GT的L1，用于观察渐进效果
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
