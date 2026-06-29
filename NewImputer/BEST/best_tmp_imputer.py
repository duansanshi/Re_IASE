import torch.nn as nn
import torch

class transformer_imputer(nn.Module):
    def __init__(self, device,d_model=128,seq_len=36,feature_dim=36,pe_dim=128):
        super().__init__()
        self.device = device
        self.to(device)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=0.1,
                batch_first=False,
            ),
            num_layers=1
        )
        self.feature_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=0.1,
                batch_first=False,
            ),
            num_layers=1
        )
        self.pe_dim = pe_dim
        self.feature_dim = feature_dim
        self.pe_proj = nn.Linear(pe_dim, d_model)
        self.pe_proj1= nn.Linear(pe_dim, d_model)
        # Precompute positional embeddings for all possible positions
        positions = torch.arange(36).float().to(device)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, pe_dim, 2).float() / pe_dim).to(device)
        pe = torch.zeros(36, pe_dim).to(device)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)  # sin for even indices
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)  # cos for odd indices
        self.register_buffer("pe", pe)  # (num_features, pe_dim)
        self.feature_pe = nn.Parameter(torch.zeros(1, feature_dim, pe_dim))  # 特征位置编码

        #self.input_proj = nn.Linear(2*feature_dim,d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(2 * feature_dim, d_model),
        )
       
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, feature_dim)
        )
        self.input_proj2 = nn.Linear(seq_len, d_model)
        self.output_proj2 = nn.Linear(d_model,seq_len)

    def forward(self, x, cond_mask):
        """
        x: observed_data*cond_mask (B,K,L)
        """
        # === 时间维度处理阶段 ===
        # 保存原始输入用于后续残差连接
        original_x = x  # [B,K,L]

        # 拼接条件和掩码
        x = torch.cat([x, cond_mask], dim=1)  # [B,2*K,L]
        x = x.permute(2,0,1)  # (L,B,2*K)
        x = self.input_proj(x)  # (L,B,d_model)
        
        # 添加位置编码
        pe = self.pe[:x.size(0), :].unsqueeze(1)  # [1, L, pe_dim]
        x = x + self.pe_proj(pe)
        
        # Transformer处理（时间维度）+ 残差
        x_trans = self.transformer(x)  # (L,B,d_model)
        x_time = x_trans + x  # 残差连接1：保留原始投影特征
        #x_time = x_trans  # 直接使用Transformer输出

        # === 特征维度处理阶段 ===
        # 时间维度投影到输出空间
        x_out = self.output_proj(x_time)  # (L,B,K)
        x_out = x_out.permute(1,2,0)  # (B,K,L)
        
        # 第二次投影（特征维度） + 残差
        x_feat = self.input_proj2(x_out)  # (B,K,d_model)
        x_feat = x_feat.permute(1,0,2)  # (K,B,d_model)
        
        # 特征位置编码
        feature_pe = self.feature_pe[:, :self.feature_dim, :]  # [1, K, pe_dim]
        x_feat = x_feat + self.pe_proj1(feature_pe).permute(1,0,2)
        
        # Transformer处理（特征维度）+ 残差
        x_feat_trans = self.feature_transformer(x_feat)  # (K,B,d_model)
        x_feat_out = x_feat_trans + x_feat  # 残差连接2：保留特征投影
        #x_feat_out = x_feat_trans  # 直接使用Transformer输出
        
        # === 最终输出 ===
        x_final = self.output_proj2(x_feat_out)  # (K,B,L)
        x_final = x_final.permute(1,0,2)  # (B,K,L)
        
        # 跨模块残差连接：最终输出与原始输入结合
        x_final = x_final + original_x  # 残差连接3：保留原始观测值

        
        return x_final

    
    
    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        coeffs = None
        if True:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()
        freq_coeffs = batch["freq"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        if True:
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