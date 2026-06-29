# TransformerAE 替换 SimpleAE
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from aqi36_lightning_datamodule import AQI36_DataModule
import matplotlib.pyplot as plt
import numpy as np

class TransformerAE(nn.Module):
    """
    Transformer-based Autoencoder that produces latent z of shape (B, k, l)
    - Spatial reduction: 1x1 conv K -> k
    - Temporal reduction: Conv1d stride = downsample_factor (L -> l)
    - Transformer encoder on tokens length l with embed dim k
    - Decoder: Transformer (same config) -> ConvTranspose1d upsample -> 1x1 conv k -> K
    Notes: requires L % l == 0 for exact down/upsampling with integer factor.
    """
    def __init__(self, K, L, k, l, n_trans_layers=2, n_heads=4, ff_dim=None, device="cuda:0"):
        super().__init__()
        self.K = K
        self.L = L
        self.k = k
        self.l = l
        self.device = device

        # check divisibility
        if L % l != 0:
            raise ValueError(f"L must be divisible by l for integer downsample factor. Got L={L}, l={l}.")

        self.down_factor = L // l  # integer factor >=1

        # 1) channel projection: K -> k (per time step)
        self.ch_proj = nn.Conv1d(K, k, kernel_size=1)  # input: (B,K,L) -> (B,k,L)

        # 2) temporal downsample: reduce length L -> l using stride = down_factor
        #   choose kernel_size=3, padding=1 so length formula: floor((L + 2p - (k-1) -1)/stride +1)
        #   For Conv1d with stride down_factor and padding=1/kernel=3 we can compute expected output length.
        self.temp_down = nn.Conv1d(k, k, kernel_size=3, stride=self.down_factor, padding=1)

        # 3) Transformer encoder on tokens with d_model = k
        d_model = k
        if ff_dim is None:
            ff_dim = max(4 * d_model, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_trans_layers)

        # positional embedding for length l
        self.pos_embed = nn.Parameter(torch.zeros(1, l, d_model))  # (1, l, k)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Decoder transformer (mirror)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_dec = nn.TransformerEncoder(decoder_layer, num_layers=n_trans_layers)

        # temporal upsample: ConvTranspose1d to go from l -> L. compute output_padding to ensure exact L
        stride = self.down_factor
        kernel_size = 3
        padding = 1
        # output_length = (l - 1) * stride - 2*padding + kernel_size + output_padding
        # => output_padding = L - ((l - 1) * stride - 2*padding + kernel_size)
        base_out = (self.l - 1) * stride - 2 * padding + kernel_size
        output_padding = self.L - base_out
        if not (0 <= output_padding < stride):
            raise ValueError(f"Calculated output_padding {output_padding} not in [0, stride). Check L,l and conv params.")
        self.temp_up = nn.ConvTranspose1d(k, k, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)

        # channel projection back: k -> K
        self.ch_unproj = nn.Conv1d(k, K, kernel_size=1)

        # small initialization
        self._init_weights()

    def _init_weights(self):
        # initialize conv weights
        nn.init.kaiming_uniform_(self.ch_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.ch_unproj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.temp_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.temp_up.weight, a=math.sqrt(5))
        if self.ch_proj.bias is not None:
            nn.init.zeros_(self.ch_proj.bias)
        if self.ch_unproj.bias is not None:
            nn.init.zeros_(self.ch_unproj.bias)

    def forward(self, x):
        """
        x: (B, K, L)
        return: recon (B, K, L), z (B, k, l)
        """
        B = x.size(0)

        # channel projection
        x_proj = self.ch_proj(x)          # (B, k, L)

        # temporal downsample
        x_down = self.temp_down(x_proj)   # (B, k, l_calc)
        # x_down temporal length should equal self.l; assert to be safe
        if x_down.size(-1) != self.l:
            # small tolerance check; otherwise raise
            raise RuntimeError(f"After temp_down expected length {self.l}, got {x_down.size(-1)}.")

        # transformer expects (B, seq_len, d_model)
        tokens = x_down.permute(0, 2, 1)  # (B, l, k)
        tokens = tokens + self.pos_embed  # add pos emb
        tokens = self.transformer_enc(tokens)  # (B, l, k)

        # latent z as (B, k, l): transpose back
        z = tokens.permute(0, 2, 1)  # (B, k, l)

        # Decoder: transformer on tokens
        tokens_dec = z.permute(0, 2, 1) + self.pos_embed  # (B, l, k)
        tokens_dec = self.transformer_dec(tokens_dec)     # (B, l, k)
        dec_feat = tokens_dec.permute(0, 2, 1)            # (B, k, l)

        # temporal upsample
        up = self.temp_up(dec_feat)  # (B, k, L) expected
        if up.size(-1) != self.L:
            # safety trim/pad (should not be needed)
            up = F.interpolate(up, size=self.L, mode='linear', align_corners=False)

        # channel unproject
        recon = self.ch_unproj(up)  # (B, K, L)

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
class simpleAELightningModule(pl.LightningModule):
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
        recon, z = self.model(x)

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

    ae_model = TransformerAE(K=K, L=L, k=k, l=l, n_trans_layers=2, n_heads=1, device=device).to(device)
    lightning_model = simpleAELightningModule(ae_model)

    aqi36_dm = AQI36_DataModule(device=device)
    aqi36_dm.setup()

    trainer = pl.Trainer(max_epochs=100, accelerator="auto", devices=1)
    trainer.fit(lightning_model, aqi36_dm)
    trainer.test(lightning_model, datamodule=aqi36_dm)


 
