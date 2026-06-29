import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGating(nn.Module):
    def __init__(self, num_layers=3, feature_dim=24):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=num_layers * feature_dim, out_channels=feature_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim * 2, out_channels=num_layers, kernel_size=1),
        )

    def forward(self, skip_results):
        stacked_features = torch.cat(skip_results, dim=1)
        attention_scores = self.gate(stacked_features)
        attention_weights = F.softmax(attention_scores, dim=1)
        all_skips = torch.stack(skip_results, dim=1)
        final_pred = torch.sum(all_skips * attention_weights.unsqueeze(2), dim=1)
        return final_pred, attention_weights


class UnetBlock(nn.Module):
    def __init__(self, device, d_model=64, seq_len=24, feature_dim=170, pe_dim=128):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=4 * d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2 * d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.transformer3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.transformer4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.dp_proj1 = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
        )
        self.dp_proj2 = nn.Sequential(
            nn.Linear(4 * d_model, 4 * d_model),
            nn.LayerNorm(4 * d_model),
        )

        self.up_proj1 = nn.Sequential(
            nn.Linear(2 * d_model, 3 * d_model),
            nn.LayerNorm(3 * d_model),
        )
        self.up_proj2 = nn.Sequential(
            nn.Linear(4 * d_model, 4 * d_model),
            nn.LayerNorm(4 * d_model),
        )

    def forward(self, x):
        residual = x
        skip0 = x.clone()
        B, K, L, _ = x.size()

        x = x.reshape(B, K, L // 3, 3 * self.d_model)
        x = self.dp_proj1(x)
        skip1 = x.clone()

        x = x.reshape(B, K, L // 6, 4 * self.d_model)
        x = self.dp_proj2(x)
        skip2 = x.clone()

        x = x.permute(0, 2, 1, 3).reshape(B * (L // 6), K, 4 * self.d_model)
        x = self.transformer1(x)
        x = x.reshape(B, L // 6, K, 4 * self.d_model).permute(0, 2, 1, 3)
        x = x + skip2

        x = self.up_proj2(x)
        x = x.reshape(B, K, L // 3, 2 * self.d_model)

        x = x.permute(0, 2, 1, 3).reshape(B * (L // 3), K, -1)
        x = self.transformer2(x)
        x = x.reshape(B, L // 3, K, -1).permute(0, 2, 1, 3)
        x = x + skip1

        x = self.up_proj1(x)
        x = x.reshape(B, K, L, self.d_model)
        x = x + skip0

        x = x.reshape(B * K, L, self.d_model)
        x = self.transformer3(x)
        x = x.reshape(B, K, L, self.d_model)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * L, K, self.d_model)
        x = self.transformer4(x)
        x = x.reshape(B, L, K, self.d_model)
        x = x.permute(0, 2, 1, 3)
        return x + residual


class transformer_imputer_cascade(nn.Module):
    def __init__(self, device, d_model=64, seq_len=24, feature_dim=170, pe_dim=128, use_dynamic_gating=False):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.use_dynamic_gating = use_dynamic_gating

        self.pe_dim = pe_dim
        self.pe_proj = nn.Linear(pe_dim + 16, d_model)
        self.feature_pe = nn.Parameter(torch.zeros(1, feature_dim, 16))

        positions = torch.arange(seq_len).float().to(device)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, pe_dim, 2).float() / pe_dim).to(device)
        pe = torch.zeros(seq_len, pe_dim).to(device)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
        self.register_buffer("seq_pe", pe)

        self.input_proj = nn.Linear(2, d_model)
        self.output_proj_1 = nn.Linear(d_model, d_model)
        self.output_proj_2 = nn.Linear(d_model, 1)
        self.feedback_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
        )

        self.UnetBlock1 = UnetBlock(self.device, self.d_model, seq_len=seq_len, feature_dim=feature_dim, pe_dim=pe_dim)
        self.UnetBlock2 = UnetBlock(self.device, self.d_model, seq_len=seq_len, feature_dim=feature_dim, pe_dim=pe_dim)
        self.UnetBlock3 = UnetBlock(self.device, self.d_model, seq_len=seq_len, feature_dim=feature_dim, pe_dim=pe_dim)
        self.dynamic_gating = DynamicGating(num_layers=3, feature_dim=seq_len)
        self.to(device)

    def _predict_increment(self, x):
        delta = self.output_proj_1(x)
        delta = F.relu(delta)
        delta = self.output_proj_2(delta).squeeze(-1)
        return delta

    def forward(self, x, cond_mask, tod=None, dow=None):
        B, K, L = x.shape

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
        prev_pred = torch.zeros(B, K, L, device=x.device, dtype=x.dtype)

        x = self.UnetBlock1(x)
        x = x + pe
        layer_delta = self._predict_increment(x)
        layer_pred = prev_pred + layer_delta
        skip_results.append(layer_pred)
        prev_pred = layer_pred
        x = x + self.feedback_proj(layer_pred.unsqueeze(-1))

        x = self.UnetBlock2(x)
        x = x + pe
        layer_delta = self._predict_increment(x)
        layer_pred = prev_pred + layer_delta
        skip_results.append(layer_pred)
        prev_pred = layer_pred
        x = x + self.feedback_proj(layer_pred.unsqueeze(-1))

        x = self.UnetBlock3(x)
        x = x + pe
        layer_delta = self._predict_increment(x)
        layer_pred = prev_pred + layer_delta
        skip_results.append(layer_pred)

        if self.use_dynamic_gating:
            final_pred, _ = self.dynamic_gating(skip_results)
        else:
            final_pred = skip_results[-1]

        return final_pred, skip_results

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()

        coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
            coeffs,
            cond_mask,
        )


transformer_imputer = transformer_imputer_cascade
