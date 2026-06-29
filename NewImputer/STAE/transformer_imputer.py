import torch.nn as nn
import torch

class transformer_imputer(nn.Module):
    def __init__(self, device,d_model=24,seq_len=36,feature_dim=36,adaptive_dim=72):
        super().__init__()
        self.device = device
        self.to(device)
        self.adaptive_dim = adaptive_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model*2+72+24,
                nhead=8,
                dropout=0.1,
                batch_first=False,
            ),
            num_layers=3
        )
        self.feature_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model*2+72+24,
                nhead=8,
                dropout=0.1,
                batch_first=False,
            ),
            num_layers=3
        )
        
        self.tod_embedding = nn.Embedding(24, d_model)
      
        self.dow_embedding = nn.Embedding(7, d_model)

        self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(feature_dim,seq_len, self.adaptive_dim))
            ).to(device)

        self.input_proj = nn.Linear(4,24)

        self.output_proj = nn.Linear(d_model*2+72+24, 1)
       

    def forward(self, x, cond_mask,tod=None, dow=None):
        """
        x: observed_data*cond_mask (B,K,L)
        """
        B, K, L = x.shape
        residual = x.clone()
        x = torch.cat([
            x.unsqueeze(3),        # (B, K, L, 1)
            cond_mask.unsqueeze(3), # (B, K, L, 1)
            tod.unsqueeze(3),       # (B, K, L, 1)
            dow.unsqueeze(3)        # (B, K, L, 1)
        ], dim=3)  # 最终形状: (B, K, L, 4)
        x = self.input_proj(x)
        features = [x]
        tod_emb = self.tod_embedding(tod.long()) #(B,K,L,24)    
        dow_emb = self.dow_embedding(dow.long()) # (B,K,L,24)
        adp_emb = self.adaptive_embedding.expand(B,*self.adaptive_embedding.shape) # (B,36,36,72)
        features.append(tod_emb)
        features.append(dow_emb)
        features.append(adp_emb)  # (B,K,L,72)
        x = torch.cat(features, dim=-1)  # (B,K,L,72)
        x = x.permute(2, 0, 1, 3).reshape(L,B*K,-1) # (L,B*K,72)
        residual_time = x.clone()  # 保存时间维度的残差
        x = self.transformer(x) + residual_time  # (L,B*K,72)
        x = x.reshape(L, B, K, -1).permute(1, 2, 0, 3)  # (B,K,L,72)
        x = x.permute(1,0,2,3).reshape(K,B*L,-1) # (K,B,L,72)
        residual_feature = x.clone()  # 保存特征维度的残差
        x = self.feature_transformer(x) + residual_feature  # (K,B,L,72)
        x = x.reshape(K, B, L, -1).permute(1, 0, 2, 3) # (B,K,L,72)
        x = self.output_proj(x).squeeze(-1)  # (B,K,L)
        x = x + residual  # Add residual connection
        return x


    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        time_of_day = batch["time_of_day"].to(self.device).float()
        day_of_week = batch["day_of_week"].to(self.device).float()

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
        time_of_day = time_of_day.permute(0, 2, 1)
        day_of_week = day_of_week.permute(0, 2, 1)

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
            time_of_day,
            day_of_week,
        )