import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 复用原有的 DynamicGating 和 UnetBlock
from transformer_imputer_new import DynamicGating, UnetBlock


def drop_path(x, drop_prob: float, training: bool):
    """Per-sample stochastic depth (DropPath).
    训练时，被drop的样本保持identity；未drop的样本输出 / keep_prob 来补偿。
    评估时，直接返回原始输入。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape: (B, 1, 1, 1) 与 x (B,K,L,d) 广播
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor_(random_tensor + keep_prob)  # 0 or 1
    output = x / keep_prob * random_tensor
    return output


class transformer_imputer_sd(nn.Module):
    """带 Stochastic Depth 的 transformer_imputer。
    
    训练时以线性递增的概率随机跳过中间 UnetBlock，
    迫使每个 block 独立产生有意义的表示，起到正则化效果。
    测试时所有 block 都激活。
    """
    def __init__(self, device, d_model=64, seq_len=36, feature_dim=36, pe_dim=128,
                 drop_path_rate=0.2):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.drop_path_rate = drop_path_rate
        
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
        
        # 输入/输出投影
        self.input_proj = nn.Linear(2, d_model)
        self.output_proj_1 = nn.Linear(d_model, d_model)
        self.output_proj_2 = nn.Linear(d_model, 1)
        
        self.to(device)
        
        # 3个UnetBlock
        self.UnetBlock1 = UnetBlock(self.device, self.d_model)
        self.UnetBlock2 = UnetBlock(self.device, self.d_model)
        self.UnetBlock3 = UnetBlock(self.device, self.d_model)
        
        # 线性递增的 drop rate：block1=0, block2=rate/2, block3=rate
        num_blocks = 3
        self.drop_rates = [drop_path_rate * i / (num_blocks - 1) for i in range(num_blocks)]
        
        self.dynamic_gating = DynamicGating(num_layers=3, feature_dim=seq_len)
    
    def _apply_block_with_drop(self, block, x, pe, drop_rate):
        """对一个 UnetBlock 施加 stochastic depth。
        
        UnetBlock 内部已有残差 (return x + residual)，
        将 drop_path 应用于 block 的"增量"部分：
            output = x + drop_path(block(x) - x)
        被 drop 时 output = x（identity），不被 drop 时正常。
        """
        block_out = block(x)          # block_out = f(x) + x  (内部已含residual)
        delta = block_out - x         # delta = f(x) (block的增量贡献)
        delta = drop_path(delta, drop_rate, self.training)
        x = x + delta                 # 等价于 x + drop_path(f(x))
        x = x + pe
        return x
    
    def forward(self, x, cond_mask, tod=None, dow=None):
        B, K, L = x.shape
        ori = x.clone()
        
        x = torch.cat([
            x.unsqueeze(-1),
            cond_mask.unsqueeze(-1),
        ], dim=-1)
        
        x = self.input_proj(x)
        
        seq_pe = self.seq_pe.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)
        feature_pe = self.feature_pe.unsqueeze(2).expand(B, -1, L, -1)
        pe = torch.cat([seq_pe, feature_pe], dim=-1)
        pe = self.pe_proj(pe)
        x = x + pe
        
        skip_results = []
        blocks = [self.UnetBlock1, self.UnetBlock2, self.UnetBlock3]
        
        for i, block in enumerate(blocks):
            x = self._apply_block_with_drop(block, x, pe, self.drop_rates[i])
            
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

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
        )
