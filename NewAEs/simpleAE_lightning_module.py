import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

# ==== VAE 核心组件 ====
class VectorQuantizerEMA(nn.Module):
    """EMA 更新码本的 VQ 层（与 LD 的 VAE 一致）"""
    def __init__(self, n_embed=8192, embed_dim=4, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1/n_embed, 1/n_embed)
        
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_sum', torch.zeros(n_embed, embed_dim))
        
    def forward(self, z):
        # z: (B, L, D)
        z_flattened = z.transpose(1, 2).reshape(-1, self.embed_dim)  # (B*L, D)
        
        # 计算距离
        dist = torch.cdist(z_flattened, self.embedding.weight) ** 2
        encoding_indices = torch.argmin(dist, dim=1)  # (B*L,)
        quantized = self.embedding(encoding_indices)  # (B*L, D)
        quantized = quantized.view(z.shape).transpose(1, 2)  # (B, L, D)
        
        # VQ Loss
        loss_vq = F.mse_loss(quantized, z)
        
        # EMA 更新码本
        if self.training:
            one_hot = F.one_hot(encoding_indices, self.n_embed).float()
            self.cluster_size = self.decay * self.cluster_size + (1 - self.decay) * one_hot.sum(0)
            embed_sum = one_hot.transpose(0, 1) @ z_flattened
            self.embed_sum = self.decay * self.embed_sum + (1 - self.decay) * embed_sum
            self.embedding.weight.data = self.embed_sum / (self.cluster_size.unsqueeze(1) + self.eps)
        
        # Straight-Through
        quantized = z + (quantized - z).detach()
        return quantized, encoding_indices, loss_vq * self.beta

class ResidualLayer(nn.Module):
    """Residual Block（LD VAE 风格）"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim*2, 3, padding=1),
            nn.GroupNorm(8, dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim*2, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.block(x)

class AttnBlock(nn.Module):
    """Self-Attention Block（LD VAE 风格）"""
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: (B, L, D)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

# ==== LD-VAE 主模型 ====
class LDVAE(nn.Module):
    def __init__(self, K=36, L=36, latent_dim=4, latent_len=36, n_embed=8192, device="cuda:0"):
        super().__init__()
        self.device = device
        self.K = K
        self.L = L
        self.latent_dim = latent_dim
        self.latent_len = latent_len
        
        # === Encoder: (B, K, L) -> (B, latent_len, latent_dim) ===
        self.encoder = nn.Sequential(
            # Downsample to latent_len
            nn.Conv1d(K, 128, 7, stride=1, padding=3),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            
            ResidualLayer(128),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),  # L/2
            nn.GroupNorm(8, 256),
            nn.GELU(),
            
            ResidualLayer(256),
            AttnBlock(256, n_heads=8),
            nn.Conv1d(256, 512, 4, stride=2, padding=1),  # L/4
            nn.GroupNorm(8, 512),
            nn.GELU(),
            
            ResidualLayer(512),
            nn.Conv1d(512, latent_dim, 3, stride=1, padding=1),  # latent_len
            nn.GroupNorm(8, latent_dim)
        )
        
        # === VQ Layer ===
        self.vq = VectorQuantizerEMA(n_embed, latent_dim)
        
        # === Decoder: (B, latent_len, latent_dim) -> (B, K, L) ===
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            ResidualLayer(512),
            
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ResidualLayer(256),
            AttnBlock(256, n_heads=8),
            
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResidualLayer(128),
            
            nn.Conv1d(128, K, 7, padding=3)
        )
        
        # === KL 正则化（可选，与标准 VAE 兼容） ===
        self.kl_scale = 1e-4
        
    def encode(self, x):
        # x: (B, K, L)
        z = self.encoder(x)  # (B, latent_dim, latent_len)
        z = z.transpose(1, 2)  # (B, latent_len, latent_dim)
        quantized, indices, vq_loss = self.vq(z)
        return quantized, indices, vq_loss
    
    def decode(self, z):
        # z: (B, latent_len, latent_dim)
        z = z.transpose(1, 2)  # (B, latent_dim, latent_len)
        recon = self.decoder(z)  # (B, K, L)
        return recon
    
    def forward(self, x):
        quantized, indices, vq_loss = self.encode(x)
        recon = self.decode(quantized)
        return recon, quantized, indices, vq_loss
    
    def process_data(self, batch):
        """与 SimpleAE 完全兼容的数据处理"""
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        coeffs = None
        if "coeffs" in batch:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()
        if "freq" in batch:
            freq_coeffs = batch["freq"].to(self.device).float()
        else:
            freq_coeffs = None

        # 转置到 (B, K, L)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        if coeffs is not None:
            coeffs = coeffs.permute(0, 2, 1)
        if freq_coeffs is not None:
            freq_coeffs = freq_coeffs.permute(0, 2, 1)

        return (
            observed_data, observed_mask, observed_tp, gt_mask,
            for_pattern_mask, cut_length, coeffs, cond_mask, freq_coeffs
        )

# ==== Lightning 模块（与 SimpleAE 完全兼容） ====
class LDVAELightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.recon_loss_fn = nn.MSELoss(reduction='none')
        
    def training_step(self, batch, batch_idx):
        data_tuple = self.model.process_data(batch)
        observed_data, observed_mask, *_ = data_tuple
        
        recon, quantized, indices, vq_loss = self.model(observed_data)
        
        # 重建损失（只在观测位置计算）
        recon_loss = self.recon_loss_fn(recon, observed_data)
        recon_loss = (recon_loss * observed_mask).sum() / observed_mask.sum()
        
        # 总损失
        loss = recon_loss + vq_loss
        
        self.log_dict({
            'train_recon_loss': recon_loss,
            'train_vq_loss': vq_loss,
            'train_total_loss': loss
        }, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data_tuple = self.model.process_data(batch)
        observed_data, observed_mask, *_ = data_tuple
        
        recon, quantized, indices, vq_loss = self.model(observed_data)
        
        recon_loss = self.recon_loss_fn(recon, observed_data)
        recon_loss = (recon_loss * observed_mask).sum() / observed_mask.sum()
        loss = recon_loss + vq_loss
        
        self.log_dict({
            'val_recon_loss': recon_loss,
            'val_vq_loss': vq_loss,
            'val_total_loss': loss
        }, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        data_tuple = self.model.process_data(batch)
        observed_data, observed_mask, *_ = data_tuple
        
        recon, quantized, indices, vq_loss = self.model(observed_data)
        
        recon_loss = self.recon_loss_fn(recon, observed_data)
        recon_loss = (recon_loss * observed_mask).sum() / observed_mask.sum()
        
        self.log('test_loss', recon_loss, prog_bar=True)
        
        # 可视化第一个样本
        if batch_idx == 0:
            self.plot_reconstruction(observed_data[0], recon[0], observed_mask[0], 
                                   f"LDVAE_test_recon_batch_{batch_idx}.png")
        
        return recon_loss
    
    def plot_reconstruction(self, x, x_hat, mask, save_path):
        """可视化重建结果"""
        x = x.detach().cpu().numpy()  # (K, L)
        x_hat = x_hat.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        
        num_sensors = min(6, x.shape[0])
        fig, axes = plt.subplots(num_sensors, 1, figsize=(12, 4*num_sensors))
        if num_sensors == 1:
            axes = [axes]
        
        for i in range(num_sensors):
            axes[i].plot(x[i], 'b-', label='Observed', linewidth=2, alpha=0.8)
            axes[i].plot(x_hat[i], 'r--', label='Reconstructed', linewidth=2, alpha=0.8)
            axes[i].fill_between(range(len(x[i])), x[i], alpha=0.3, color='blue')
            axes[i].set_title(f'Sensor {i+1} (Masked: {mask[i].sum()/len(mask[i]):.1%})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]

# ==== 使用示例（完全兼容你的框架） ====
if __name__ == "__main__":
    import pytorch_lightning as pl
    from aqi36_lightning_datamodule import AQI36_DataModule
    
    # === 超参数 ===
    K, L = 36, 36
    latent_dim, latent_len = 4, 36  # LD 风格的紧凑隐空间
    
    # === 模型初始化 ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_model = LDVAE(K=K, L=L, latent_dim=latent_dim, latent_len=latent_len).to(device)
    lightning_model = LDVAELightningModule(vae_model)
    
    # === 数据模块（完全兼容） ===
    aqi36_dm = AQI36_DataModule(device=device)
    aqi36_dm.setup()
    
    # === 训练 ===
    trainer = pl.Trainer(
        max_epochs=50, 
        accelerator="auto", 
        devices=1,
        precision=16,  # 混合精度
        logger=pl.loggers.TensorBoardLogger("ldvae_logs/"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_total_loss', save_top_k=3),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ]
    )
    
    trainer.fit(lightning_model, aqi36_dm)
    
    # === 测试 ===
    trainer.test(lightning_model, aqi36_dm)
    
    print("✅ LD-VAE 训练完成！")
    print(f"   - Latent shape: ({latent_dim}, {latent_len})")
    print(f"   - Compression ratio: {K*L/(latent_dim*latent_len):.1f}x")