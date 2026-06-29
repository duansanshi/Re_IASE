import torch.nn as nn
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

######################
# 可学习频率分解模块
######################

class AdaptiveFreqDecomposition(nn.Module):
    """
    可学习的频率分解模块，灵感来自 Autoformer / FEDformer / FilterNet。
    在频域中使用可学习的软截断滤波器，将输入分解为 trend（低频趋势）和 detail（高频残差）。
    与固定 cutoff 的 FFT 硬截断不同，这里的滤波器完全可学习。
    """
    def __init__(self, seq_len):
        super().__init__()
        freq_len = seq_len // 2 + 1  # rfft 输出长度
        # 初始化：低频偏向高通过率，高频偏向低通过率
        # sigmoid(2)≈0.88, sigmoid(-2)≈0.12
        init_filter = torch.linspace(2, -2, freq_len)
        self.freq_filter = nn.Parameter(init_filter)
        
    def forward(self, x):
        """
        x: (B, K, L) — 时域输入
        返回: trend (B, K, L), detail (B, K, L)
        """
        X_freq = torch.fft.rfft(x, dim=-1)  # (B, K, freq_len)
        low_pass = torch.sigmoid(self.freq_filter)  # (freq_len,) 可学习软掩码
        
        trend = torch.fft.irfft(X_freq * low_pass, n=x.shape[-1], dim=-1)
        detail = torch.fft.irfft(X_freq * (1 - low_pass), n=x.shape[-1], dim=-1)
        
        return trend, detail


######################
# DynamicGating (保持兼容)
######################

class DynamicGating(nn.Module):
    def __init__(self, num_layers=3, feature_dim=36):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=num_layers * feature_dim, out_channels=feature_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim * 2, out_channels=num_layers, kernel_size=1)
        )

    def forward(self, skip_results):
        stacked_features = torch.cat(skip_results, dim=1)
        attention_scores = self.gate(stacked_features)
        attention_weights = F.softmax(attention_scores, dim=1)
        all_skips = torch.stack(skip_results, dim=1)
        attention_weights_expanded = attention_weights.unsqueeze(2)
        final_pred = torch.sum(all_skips * attention_weights_expanded, dim=1)
        return final_pred, attention_weights


######################
# UnetBlock (与 v5 完全相同)
######################

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class UnetBlock(nn.Module):
    def __init__(self, device, d_model=64, seq_len=36, feature_dim=36, pe_dim=128):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim

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
            nn.Linear(3*d_model, 2*d_model),
            nn.LayerNorm(2*d_model)
        )
        self.dp_proj2 = nn.Sequential(
            nn.Linear(8*d_model, 4*d_model),
            nn.LayerNorm(4*d_model)
        )
        self.up_proj1 = nn.Sequential(
            nn.Linear(2*d_model, 3*d_model),
            nn.LayerNorm(3*d_model)
        )
        self.up_proj2 = nn.Sequential(
            nn.Linear(4*d_model, 8*d_model),
            nn.LayerNorm(8*d_model)
        )
 
    def forward(self, x):
        residual = x
        skip0 = x.clone()
        B, K, L, _ = x.size()
        
        x = x.reshape(B, K, L//3, 3*self.d_model)
        x = self.dp_proj1(x)
        skip1 = x.clone()
        
        x = x.reshape(B, K, L//12, 8*self.d_model)
        x = self.dp_proj2(x)
        skip2 = x.clone()
        
        x = x.permute(0,2,1,3).reshape(B*L//12, K, -1)
        x = self.transformer1(x)
        x = x.reshape(B, L//12, K, -1).permute(0,2,1,3)
        x = x + skip2
        
        x = self.up_proj2(x)
        x = x.reshape(B, K, L//3, 2*self.d_model)
        
        x = x.permute(0,2,1,3).reshape(B*L//3, K, -1)
        x = self.transformer2(x)
        x = x.reshape(B, L//3, K, -1).permute(0,2,1,3)
        x = x + skip1
        
        x = self.up_proj1(x)
        x = x.reshape(B, K, L, self.d_model)
        x = x + skip0
        
        x = x.reshape(B*K, L, self.d_model)
        x = self.transformer3(x)
        x = x.reshape(B, K, L, self.d_model)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B*L, K, self.d_model)
        x = self.transformer4(x)
        x = x.reshape(B, L, K, self.d_model)
        x = x.permute(0, 2, 1, 3)
        
        return x + residual


######################
# v9 主模型：可学习频率分解 + 4通道输入
######################

class transformer_imputer(nn.Module):
    def __init__(self, device, d_model=64, seq_len=36, feature_dim=36, pe_dim=128):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # 可学习频率分解
        self.decomposition = AdaptiveFreqDecomposition(seq_len)
        
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
      
        # 输入投影层：4通道 [raw, mask, trend, detail]
        self.input_proj = nn.Linear(4, d_model)
        
        # 输出层（共享）
        self.output_proj_1 = nn.Linear(d_model, d_model)
        self.output_proj_2 = nn.Linear(d_model, 1)
        
        self.to(device)

        self.UnetBlock1 = UnetBlock(self.device, self.d_model).to(device)
        self.UnetBlock2 = UnetBlock(self.device, self.d_model).to(device)
        self.UnetBlock3 = UnetBlock(self.device, self.d_model).to(device)
        
        self.dynamic_gating = DynamicGating(num_layers=3, feature_dim=seq_len)

    def forward(self, x, cond_mask, tod=None, dow=None):
        """
        x: coeffs * cond_mask (B, K, L) — 预填充后的观测值
        cond_mask: 缺失值掩码 (B, K, L)
        """
        B, K, L = x.shape
        
        # ====== 可学习频率分解 ======
        trend, detail = self.decomposition(x)  # 各 (B, K, L)
        
        # ====== 4通道输入拼接 ======
        x = torch.cat([
            x.unsqueeze(-1),          # raw         (B, K, L, 1)
            cond_mask.unsqueeze(-1),   # mask        (B, K, L, 1)
            trend.unsqueeze(-1),       # low-pass    (B, K, L, 1)
            detail.unsqueeze(-1),      # high-pass   (B, K, L, 1)
        ], dim=-1)  # (B, K, L, 4)
        
        # 输入投影
        x = self.input_proj(x)  # (B, K, L, d_model)
  
        # 位置编码
        seq_pe = self.seq_pe.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)
        feature_pe = self.feature_pe.unsqueeze(2).expand(B, -1, L, -1)
        pe = torch.cat([seq_pe, feature_pe], dim=-1)
        pe = self.pe_proj(pe)
        x = x + pe
        
        # ====== 3个 UnetBlock + 深度监督 ======
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
        
        # 最终输出
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

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )
