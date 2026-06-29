
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# 基础组件
# ------------------------------
def get_torch_trans(heads: int = 8, layers: int = 1, channels: int = 64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


# ------------------------------
# （保留，不再使用）扩散步嵌入
# ------------------------------
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim: int = 128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    @staticmethod
    def _build_embedding(num_steps, dim: int = 64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = (10.0 ** (torch.arange(dim) / (dim - 1) * 4.0)).unsqueeze(0)  # (1,dim)
        table = steps * frequencies                                                     # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)                 # (T,2·dim)
        return table

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x


# ------------------------------
# 主网络
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, side_dim: int, channels: int, nheads: int, is_linear: bool = False):
        super().__init__()

        # 已去除 diffusion_projection
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    # ---- 时间维 Transformer ----
    def forward_time(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (B·K, C, L)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)
        return y

    # ---- 特征维 Transformer ----
    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (B·L, C, K)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)
        return y

    # ---- 前向 ----
    def forward(self, x, cond_info):
        B, C, K, L = x.shape
        base_shape = x.shape
        x_flat = x.reshape(B, C, K * L)

        y = x_flat  # 不再加 diffusion_emb
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)  # (B, 2C, K·L)

        cond = cond_info.reshape(B, cond_info.size(1), K * L)
        cond = self.cond_projection(cond)  # (B, 2C, K·L)
        y = y + cond

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B, C, K·L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip


class diff_CSDI(nn.Module):
    def __init__(self, device, inputdim: int = 2):
        super().__init__()
        self.device = device
        self.channels = 64

        # ---- 嵌入 ----
        self.embed_layer = nn.Embedding(num_embeddings=36, embedding_dim=16)

        # ---- 输入 / 输出投影 ----
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        # ---- 残差块 ----
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=128 + 16 + 1,  # time_emb + feature_emb + cond_mask
                    channels=self.channels,
                    nheads=8,
                    is_linear=False,
                )
                for _ in range(4)
            ]
        )

    # ---- 正弦时间嵌入 ----
    def time_embedding(self, pos, d_model: int = 128):
        pe = torch.zeros(pos.size(0), pos.size(1), d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1.0 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=self.device) / d_model
        )
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe

    # ---- side info 组装 ----
    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, 128).unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(36, device=self.device))  # (K,16)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,144+)
        side_info = side_info.permute(0, 3, 2, 1)                   # (B,*,K,L)
        side_mask = cond_mask.unsqueeze(1)                          # (B,1,K,L)

        return torch.cat([side_info, side_mask], dim=1)             # (B,128+16+1,K,L)

    # ---- 前向 ----
    def forward(self, x, cond_info, diffusion_step=None):
        """
        x          : (B, inputdim, K, L)
        cond_info  : (B, 145, K, L)  # 由 get_side_info 返回
        """
        B, D, K, L = x.shape
        x = x.reshape(B, D, K * L)
        x = F.relu(self.input_projection(x))
        x = x.reshape(B, self.channels, K, L)

        skip_connections = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_info)
            skip_connections.append(skip)

        x = torch.sum(torch.stack(skip_connections), dim=0) / math.sqrt(len(self.residual_layers))
        x = F.relu(self.output_projection1(x.reshape(B, self.channels, K * L)))
        x = self.output_projection2(x).reshape(B, K, L)
        return x

    # ---- 数据预处理（与原实现保持一致）----
    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        coeffs = batch["coeffs"].to(self.device).float() if "coeffs" in batch else None
        cond_mask = batch["cond_mask"].to(self.device).float()

        # (B, K, L)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        if coeffs is not None:
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

