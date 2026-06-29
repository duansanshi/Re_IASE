"""v7: 方案B - 频率辅助输入（双编码器加法融合）
   input_proj: Linear(2, d_model) ← [raw_preimputed, mask]
   freq_proj:  Linear(1, d_model) ← [freq_filtered]
   x = input_proj(raw, mask) + freq_proj(freq)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class UnetBlock(nn.Module):
    def __init__(self, device, d_model=64, seq_len=36, feature_dim=36, pe_dim=128):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # Transformer编码器
        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=4*d_model, nhead=8, dropout=0.1,
                batch_first=True, norm_first=True,
            ), num_layers=1
        )
        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2*d_model, nhead=8, dropout=0.1,
                batch_first=True, norm_first=True,
            ), num_layers=1
        )
        self.transformer3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dropout=0.1,
                batch_first=True, norm_first=True,
            ), num_layers=1
        )
        self.transformer4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dropout=0.1,
                batch_first=True, norm_first=True,
            ), num_layers=1
        )

        self.dp_proj1 = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model)
        )
        self.dp_proj2 = nn.Sequential(
            nn.Linear(8 * d_model, 4 * d_model),
            nn.LayerNorm(4 * d_model)
        )
        self.up_proj1 = nn.Sequential(
            nn.Linear(2 * d_model, 3 * d_model),
            nn.LayerNorm(3 * d_model)
        )
        self.up_proj2 = nn.Sequential(
            nn.Linear(4 * d_model, 8 * d_model),
            nn.LayerNorm(8 * d_model)
        )

    def forward(self, x):
        """
        feature (B,K,L,d_model) --> same size
        """
        residual = x
        skip0 = x.clone()
        B, K, L, _ = x.size()

        # --- 下采样 ---
        x = x.reshape(B, K, L // 3, 3 * self.d_model)
        x = self.dp_proj1(x)
        skip1 = x.clone()

        x = x.reshape(B, K, L // 12, 8 * self.d_model)
        x = self.dp_proj2(x)
        skip2 = x.clone()

        # --- 空间 Transformer ---
        x = x.permute(0, 2, 1, 3).reshape(B * L // 12, K, -1)
        x = self.transformer1(x)
        x = x.reshape(B, L // 12, K, -1).permute(0, 2, 1, 3)
        x = x + skip2

        # --- 上采样 ---
        x = self.up_proj2(x)
        x = x.reshape(B, K, L // 3, 2 * self.d_model)

        x = x.permute(0, 2, 1, 3).reshape(B * L // 3, K, -1)
        x = self.transformer2(x)
        x = x.reshape(B, L // 3, K, -1).permute(0, 2, 1, 3)
        x = x + skip1

        x = self.up_proj1(x)
        x = x.reshape(B, K, L, self.d_model)
        x = x + skip0

        # --- 双路径 Transformer ---
        x = x.reshape(B * K, L, self.d_model)
        x = self.transformer3(x)
        x = x.reshape(B, K, L, self.d_model)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * L, K, self.d_model)
        x = self.transformer4(x)
        x = x.reshape(B, L, K, self.d_model)
        x = x.permute(0, 2, 1, 3)

        return x + residual


class transformer_imputer(nn.Module):
    def __init__(self, device, d_model=64, seq_len=36, feature_dim=36, pe_dim=128):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # 位置编码
        self.pe_dim = pe_dim
        self.pe_proj = nn.Linear(pe_dim + 16, d_model)
        self.feature_pe = nn.Parameter(torch.zeros(1, feature_dim, 16))

        positions = torch.arange(seq_len).float().to(device)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, pe_dim, 2).float() / pe_dim).to(device)
        pe = torch.zeros(seq_len, pe_dim).to(device)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
        self.register_buffer("seq_pe", pe)

        # 原始输入投影: [raw, mask] → d_model
        self.input_proj = nn.Linear(2, d_model)
        # 频率辅助投影: [freq] → d_model
        self.freq_proj = nn.Linear(1, d_model)
        # 共享输出头
        self.output_proj_1 = nn.Linear(d_model, d_model)
        self.output_proj_2 = nn.Linear(d_model, 1)

        self.to(device)

        self.UnetBlock1 = UnetBlock(self.device, self.d_model).to(device)
        self.UnetBlock2 = UnetBlock(self.device, self.d_model).to(device)
        self.UnetBlock3 = UnetBlock(self.device, self.d_model).to(device)

    def forward(self, x, cond_mask, freq_coeffs=None, tod=None, dow=None):
        """
        x: coeffs (B, K, L) - 原始预填充数据
        cond_mask: (B, K, L) 1=observed, 0=missing
        freq_coeffs: (B, K, L) - 频率滤波后的辅助数据
        """
        B, K, L = x.shape
        if freq_coeffs is None:
            freq_coeffs = torch.zeros_like(x)

        # 原始输入编码
        x_main = torch.cat([
            x.unsqueeze(-1),
            cond_mask.unsqueeze(-1),
        ], dim=-1)  # (B, K, L, 2)
        x_main = self.input_proj(x_main)  # (B, K, L, d_model)

        # 频率辅助编码
        x_freq = freq_coeffs.unsqueeze(-1)  # (B, K, L, 1)
        x_freq = self.freq_proj(x_freq)  # (B, K, L, d_model)

        # 加法融合
        x = x_main + x_freq  # (B, K, L, d_model)

        # 位置编码
        seq_pe = self.seq_pe.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)
        feature_pe = self.feature_pe.unsqueeze(2).expand(B, -1, L, -1)
        pe = torch.cat([seq_pe, feature_pe], dim=-1)
        pe = self.pe_proj(pe)
        x = x + pe

        skip_results = []

        # Block 1
        x = self.UnetBlock1(x)
        x = x + pe
        layer_pred = self.output_proj_1(x)
        layer_pred = F.relu(layer_pred)
        layer_pred = self.output_proj_2(layer_pred).squeeze(-1)
        skip_results.append(layer_pred)

        # Block 2
        x = self.UnetBlock2(x)
        x = x + pe
        layer_pred = self.output_proj_1(x)
        layer_pred = F.relu(layer_pred)
        layer_pred = self.output_proj_2(layer_pred).squeeze(-1)
        skip_results.append(layer_pred)

        # Block 3
        x = self.UnetBlock3(x)
        x = x + pe
        layer_pred = self.output_proj_1(x)
        layer_pred = F.relu(layer_pred)
        layer_pred = self.output_proj_2(layer_pred).squeeze(-1)
        skip_results.append(layer_pred)

        final_pred = skip_results[-1]
        return final_pred, skip_results

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()
        freq_coeffs = batch.get("freq_coeffs")
        if freq_coeffs is not None:
            freq_coeffs = freq_coeffs.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        coeffs = coeffs.permute(0, 2, 1)
        if freq_coeffs is not None:
            freq_coeffs = freq_coeffs.permute(0, 2, 1)

        return (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask, freq_coeffs,
        )
