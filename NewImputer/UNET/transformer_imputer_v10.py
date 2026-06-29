import torch.nn as nn
import torch
import math
import torch.nn.functional as F

######################
# Gap-Length Prompting: 缺失长度感知
######################

def compute_gap_lengths(mask):
    """
    从 cond_mask 中计算每个缺失位置所属连续缺失区间的长度。
    mask: (B, K, L) — 1=observed, 0=missing
    返回: gap_lengths (B, K, L) — missing位置=所属gap的总长度, observed位置=0
    """
    B, K, L = mask.shape
    missing = (1 - mask).long()  # 1 where missing
    
    # Forward: 从左到右累计连续缺失长度
    fwd = torch.zeros(B, K, L, dtype=torch.long, device=mask.device)
    for t in range(L):
        if t == 0:
            fwd[:, :, t] = missing[:, :, t]
        else:
            fwd[:, :, t] = (fwd[:, :, t-1] + 1) * missing[:, :, t]
    
    # Backward: 从右到左累计连续缺失长度
    bwd = torch.zeros(B, K, L, dtype=torch.long, device=mask.device)
    for t in range(L-1, -1, -1):
        if t == L-1:
            bwd[:, :, t] = missing[:, :, t]
        else:
            bwd[:, :, t] = (bwd[:, :, t+1] + 1) * missing[:, :, t]
    
    # gap_length = fwd + bwd - 1 (对于missing位置)
    gap_len = (fwd + bwd - 1) * missing  # observed位置=0
    
    return gap_len


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
# v10 主模型：Gap-Length Prompting
# 输入: [raw(1), mask(1), gap_embed(gap_embed_dim)] → d_model
######################

class transformer_imputer(nn.Module):
    def __init__(self, device, d_model=64, seq_len=36, feature_dim=36, pe_dim=128,
                 max_gap=36, gap_embed_dim=16):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.max_gap = max_gap
        
        # Gap-Length Embedding: 0=observed, 1..max_gap=缺失长度
        self.gap_embedding = nn.Embedding(max_gap + 1, gap_embed_dim)
        
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
      
        # 输入投影层：[raw(1) + mask(1) + gap_embed(gap_embed_dim)]
        self.input_proj = nn.Linear(2 + gap_embed_dim, d_model)
        
        # 输出层
        self.output_proj_1 = nn.Linear(d_model, d_model)
        self.output_proj_2 = nn.Linear(d_model, 1)
        
        self.to(device)

        self.UnetBlock1 = UnetBlock(self.device, self.d_model).to(device)
        self.UnetBlock2 = UnetBlock(self.device, self.d_model).to(device)
        self.UnetBlock3 = UnetBlock(self.device, self.d_model).to(device)
        
        self.dynamic_gating = DynamicGating(num_layers=3, feature_dim=seq_len)

    def forward(self, x, cond_mask, tod=None, dow=None):
        """
        x: coeffs * cond_mask (B, K, L)
        cond_mask: 缺失值掩码 (B, K, L), 1=observed, 0=missing
        """
        B, K, L = x.shape
        
        # ====== 计算 Gap 长度 ======
        gap_len = compute_gap_lengths(cond_mask)  # (B, K, L), long
        gap_len = gap_len.clamp(0, self.max_gap)  # 截断到 max_gap
        gap_embed = self.gap_embedding(gap_len)    # (B, K, L, gap_embed_dim)
        
        # ====== 输入拼接: [raw, mask, gap_embed] ======
        x = torch.cat([
            x.unsqueeze(-1),          # raw         (B, K, L, 1)
            cond_mask.unsqueeze(-1),   # mask        (B, K, L, 1)
            gap_embed,                 # gap prompt  (B, K, L, gap_embed_dim)
        ], dim=-1)  # (B, K, L, 2 + gap_embed_dim)
        
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
