import torch.nn as nn
import torch
import torch.nn.functional as F
from mamba_ssm import Mamba

class transformer_imputer(nn.Module):
    def __init__(self, device, input_dim=36, scales=[3,6,9,12,18], stride=2,d_model=128):
        super().__init__()
        self.device = device
        self.input_dim = input_dim  # Input feature dimension K
        self.scales = scales        # Pooling scale list, e.g., [3, 5, 7]
        self.stride = stride        # Pooling stride to reduce time dimension
        self.d_model = d_model
        
        # Define multi-scale pooling layers
        self.pool_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=s, stride=s, padding=0) for s in scales
        ])
        
        # Projection layer to map concatenated features back to original dimension
        self.projection = nn.Conv1d(input_dim * (len(scales) + 2), input_dim, kernel_size=1)
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=d_model,
        #         nhead=8,
        #         dropout=0.1,
        #         batch_first=False,
        #     ),
        #     num_layers=1
        # )
        # self.output_proj = nn.Conv1d(d_model, input_dim, kernel_size=1)
    
    def forward(self, x, cond_mask):
        # x: (B, K, L)
        # cond_mask: (B, K, L)
        B, K, L = x.shape
        
        # Store features for concatenation
        features = [x, cond_mask]
        
        # Apply multi-scale pooling to input x
        for pool in self.pool_layers:
            pooled = pool(x)  # Shape becomes (B, K, T_i) where T_i < L
            # Interpolate back to original length L
            pooled = F.interpolate(pooled, size=L, mode='linear', align_corners=False)
            features.append(pooled)
        
        # Concatenate all features along the feature dimension
        multi_scale_features = torch.cat(features, dim=1)  # (B, K * (n + 2), L)
        
        # Project back to original dimension
        output = self.projection(multi_scale_features)  # (L,B,d_model)
        # Apply transformer
        # output = self.transformer(output) # (L,B,d_model)
        # # Project back to original dimension
        # output = self.output_proj(output.permute(1,2,0))
        
        return output

    
# class transformer_imputer(nn.Module):
#     def __init__(self, device, input_dim=36, scales=[3, 5, 7,9,11,13], stride=2):
#         super().__init__()
#         self.device = device
#         self.input_dim = input_dim  # Input feature dimension K
#         self.scales = scales        # Convolution kernel sizes, e.g., [3, 5, 7]
#         self.stride = stride        # Convolution stride
        
#         # Define multi-scale convolutional layers
#         self.conv_layers = nn.ModuleList([
#             nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=s, stride=stride, padding=0)
#             for s in scales
#         ])
        
#         # Projection layer to map concatenated features back to original dimension
#         self.projection = nn.Conv1d(input_dim * (len(scales) + 2), input_dim, kernel_size=1)
    
#     def forward(self, x, cond_mask):
#         # x: (B, K, L)
#         # cond_mask: (B, K, L)
#         B, K, L = x.shape
        
#         # Store features for concatenation
#         features = [x, cond_mask]
        
#         # Apply multi-scale convolution to input x
#         for conv in self.conv_layers:
#             conved = conv(x)  # Shape becomes (B, K, T_i) where T_i < L
#             # Interpolate back to original length L
#             conved = F.interpolate(conved, size=L, mode='linear', align_corners=False)
#             features.append(conved)
        
#         # Concatenate all features along the feature dimension
#         multi_scale_features = torch.cat(features, dim=1)  # (B, K * (n + 2), L)
        
#         # Project back to original dimension
#         output = self.projection(multi_scale_features)  # (B, K, L)
        
#         return output
    
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