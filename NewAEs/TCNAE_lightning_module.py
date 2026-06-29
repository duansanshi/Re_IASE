import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from aqi36_lightning_datamodule import AQI36_DataModule
import matplotlib.pyplot as plt
import numpy as np

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)

class TCNAE(nn.Module):
    """
    Temporal Convolutional Autoencoder (TCN-AE)
    结构：
      Encoder: Conv1d 堆叠 + stride 降采样
      Decoder: ConvTranspose1d 上采样 + Residual 块
    """
    def __init__(self, K, L, k, l, device="cuda:0"):
        super().__init__()
        self.K, self.L, self.k, self.l = K, L, k, l
        self.device = device

        down_factor = L // l
        assert L % l == 0, "L must be divisible by l."

        # ===== Encoder =====
        self.encoder = nn.Sequential(
            ResidualTCNBlock(K, 64, dilation=1),
            ResidualTCNBlock(64, 128, dilation=2),
            nn.Conv1d(128, k, kernel_size=3, stride=down_factor, padding=1),
            nn.ReLU(),
        )

        # ===== Decoder =====
        # ConvTranspose1d 恢复时间长度
        output_padding = (L - ((l - 1) * down_factor - 2 * 1 + 3))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(k, 128, kernel_size=3, stride=down_factor, padding=1, output_padding=output_padding),
            nn.ReLU(),
            ResidualTCNBlock(128, 64, dilation=1),
            nn.Conv1d(64, K, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        x: (B, K, L)
        return: recon (B, K, L), z (B, k, l)
        """
        z = self.encoder(x)   # (B, k, l)
        recon = self.decoder(z)  # (B, K, L)
        return recon, z

    def process_data(self, batch):
        """
        Keep same API as original SimpleAE.process_data so LightningModule can call it.
        Assumes batch is a dict with keys used earlier.
        """
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        coeffs = None
        if "coeffs" in batch:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch.get("cond_mask", torch.ones_like(observed_mask)).to(self.device).float()
        freq_coeffs = batch.get("freq", torch.zeros_like(observed_mask)).to(self.device).float()

        # permute to (B, K, L)
        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        if coeffs is not None:
            coeffs = coeffs.permute(0, 2, 1)
        freq_coeffs = freq_coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            freq_coeffs,
        )

# ==== Lightning 模块 ====
class TCNAELightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # 模仿实际数据流程
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
        ) = self.model.process_data(batch)

        # 假设我们要对 observed_data 进行AE压缩
        x = observed_data
        recon, z = self.model(x)s

        loss = self.loss_fn(recon * observed_mask, x * observed_mask)
        self.log("train_loss", loss)
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
            freq_coeffs,
        ) = self.model.process_data(batch)

        x = observed_data
        recon, z = self.model(x)

        loss = self.loss_fn(recon * observed_mask, x * observed_mask)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        observed_data, observed_mask, *_ = self.model.process_data(batch)
        recon, z = self.model(observed_data)
        loss = ((recon - observed_data)**2 * observed_mask).sum() / observed_mask.sum()
        self.log("test_loss", loss, prog_bar=True)
        # 选第一个样本
        x = observed_data[0].detach().cpu().numpy()
        x_hat = recon[0].detach().cpu().numpy()

        # 假设 shape = [num_sensors, time_len]
        num_sensors = min(3, x.shape[0])  # 只画前三个传感器
        fig, axes = plt.subplots(num_sensors, 1, figsize=(8, 6))
        if num_sensors == 1:
            axes = [axes]

        for i in range(num_sensors):
            axes[i].plot(x[i], label='Observed', linewidth=1.5)
            axes[i].plot(x_hat[i], label='Reconstructed', linestyle='--', linewidth=1.2)
            axes[i].set_title(f"Sensor {i+1}")
            axes[i].legend()

        plt.tight_layout()
        plt.savefig("NewAEs/test.png")

        return {'loss': loss, 'recon': recon, 'x': observed_data}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        # 余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,       # 一个周期的迭代次数（可设置为总训练步数或轮数）
            eta_min=1e-5    # 最低学习率
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # 每轮更新一次
                "frequency": 1
            }
        }



if __name__ == "__main__":
    K, L = 36, 36
    k, l = 18, 18   # 注意：l 必须整除 L (36 % 18 == 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ae_model = TCNAE(K=36, L=36, k=18, l=18, device=device).to(device)
    lightning_model = TCNAELightningModule(ae_model)


    aqi36_dm = AQI36_DataModule(device=device)
    aqi36_dm.setup()

    trainer = pl.Trainer(max_epochs=100, accelerator="auto", devices=1)
    trainer.fit(lightning_model, aqi36_dm)
    trainer.test(lightning_model, datamodule=aqi36_dm)


 

