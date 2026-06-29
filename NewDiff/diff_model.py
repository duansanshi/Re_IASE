import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from generate_adj import *



class AdaptiveGCN(nn.Module):
    def __init__(self, channels, order=2, include_self=True, device=None, is_adp=True, adj_file=None):
        super().__init__()
        
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 2
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1

        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x, base_shape, support_adp):
        B, channel, K, L = base_shape
        if K == 1:
            return x
        if self.is_adp:
            nodevec1 = support_adp[-1][0]
            nodevec2 = support_adp[-1][1]
            support = support_adp[:-1]
        else:
            support = support_adp
        x = x.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        if self.is_adp:
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
            support = support + [adp]
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        out = out.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return out


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table



    
class transformer_imputer(nn.Module):
    def __init__(self, device, d_model=128, seq_len=36, feature_dim=36, pe_dim=128):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        #################################
        self.GCN = AdaptiveGCN(channels=d_model,order=2,include_self=True,device=device,is_adp=True,adj_file='AQI36')
        self.adj = get_adj_AQI36()
        self.support = compute_support_gwn(self.adj, device=device)
        node_num = self.adj.shape[0]
        self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
        self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
        self.support.append([self.nodevec1, self.nodevec2])
        #################################
        # Transformer编码器
        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=4*d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2
        )

        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2*d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2
        )
     
        self.transformer3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=3
        )
        self.transformer4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=3
        )
        
        # 下采样和上采样投影层
        self.dp_proj1 = nn.Sequential(
            #nn.LayerNorm(3*d_model),
            nn.Linear(3*d_model, 2*d_model),
            nn.LayerNorm(2*d_model)
        )
        self.dp_proj2 = nn.Sequential(
            #nn.LayerNorm(8*d_model),
            nn.Linear(8*d_model, 4*d_model),
            nn.LayerNorm(4*d_model)
        )
        self.up_proj1 = nn.Sequential(
            #nn.LayerNorm(2*d_model),
            nn.Linear(2*d_model, 3*d_model),
            nn.LayerNorm(3*d_model)
        )
        self.up_proj2 = nn.Sequential(
            #nn.LayerNorm(4*d_model),
            nn.Linear(4*d_model, 8*d_model),
            nn.LayerNorm(8*d_model)
        )

        # 位置编码模块
        self.pe_dim = pe_dim
        self.pe_proj = nn.Linear(pe_dim+16, d_model)
        self.feature_pe = nn.Parameter(torch.zeros(1, feature_dim, 16))


        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=100,
            embedding_dim=128,
        )
        #self.diffusion_embedding = DiffusionEmbedding()

        self.diffusion_projection = nn.Linear(128,d_model)
        
        # 预计算序列位置编码
        positions = torch.arange(seq_len).float().to(device)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, pe_dim, 2).float() / pe_dim).to(device)
        pe = torch.zeros(seq_len, pe_dim).to(device)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
        self.register_buffer("seq_pe", pe)  # (seq_len, pe_dim)
      
        
        # 输入投影层
        self.input_proj = nn.Linear(3, d_model)  
        # 输出层
        self.output_proj_1 = nn.Linear(d_model, d_model)
        self.output_proj_2 = nn.Linear(d_model, 1)   
        self.to(device)
      
        
    def forward(self, x, cond_mask, noisy_data, diffusion_step):
        """
        x: observed_data * cond_mask (B, K, L)
        cond_mask: 缺失值掩码 (B, K, L)
        tod: 时间信息 (B, K, L)
        dow: 日期信息 (B, K, L)
        """
        B, K, L = x.shape
        ori = x.clone()  # 保存原始输入用于残差连接
        
      
        # 拼接输入特征
        x = torch.cat([
            x.unsqueeze(-1),        # (B, K, L, 1)
            cond_mask.unsqueeze(-1), # (B, K, L, 1)
            noisy_data.unsqueeze(-1), # (B, K, L, 1)
           
        ], dim=-1)  # (B, K, L, 3)
        
        # 输入投影
       
        x = self.input_proj(x)  # (B, K, L, d_model)
        ############# for GCN
        x_gcn = x.clone()
        x_gcn = x_gcn.permute(0,3,1,2) #(B,d,K,L)
        base_shape_gcn = x_gcn.shape
        x_gcn = x_gcn.reshape(base_shape_gcn[0],base_shape_gcn[1],-1)
        x_gcn = self.GCN(x_gcn,base_shape_gcn,self.support)
        x_gcn = x_gcn.reshape(base_shape_gcn[0],base_shape_gcn[1],base_shape_gcn[2],base_shape_gcn[3]) #(B,d,K,L)
        x_gcn = x_gcn.permute(0,2,3,1) #(B,K,L,d)
        #############
        x = x + x_gcn

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        diffusion_feature = self.diffusion_projection(diffusion_emb).unsqueeze(1).unsqueeze(2)
        
        x = x + diffusion_feature
  
        # 添加位置编码

        seq_pe = self.seq_pe.unsqueeze(0).unsqueeze(0).expand(B,K,-1,-1)  # (1, 1, L, pe_dim)
        feature_pe = self.feature_pe.unsqueeze(2).expand(B,-1,L,-1) 
  
        pe = torch.cat([seq_pe, feature_pe], dim=-1)  # (B, K, L, pe_dim + feature_dim)
        pe = self.pe_proj(pe)  # (B, K, L, d_model)
        x = x + pe  # (B, K, L, d_model)
       
        residual = x #for skip connnection
        # 初始残差连接
        skip0 = x.clone()
        B,K,L,_ = x.size()
        
        # --- 下采样路径 ---
        # 第一级下采样
        x = x.reshape(B, K, L//3, 3*self.d_model)  # (B, K, L/3, 3*d_model)
        x = self.dp_proj1(x)  # (B, K, L/3, 2*d_model)
        skip1 = x.clone()
        
        # 第二级下采样
        x = x.reshape(B, K, L//12, 8*self.d_model)  # (B, K, L/12, 8*d_model)
        x = self.dp_proj2(x)  # (B, K, L/12, 4*d_model)
        skip2 = x.clone()
        
        # --- Transformer处理 ---
        # 第一级Transformer
        # x = x.reshape(B*K, L//12, -1)  # (B*K, L/12, 4*d_model)
        # x = self.transformer1(x)  # (B*K, L/12, 4*d_model)
        # x = x.reshape(B, K, L//12, -1)  # (B, K, L/12, 4*d_model)

        x = x.permute(0,2,1,3).reshape(B*L//12,K,-1) # (B*L//12, K, 4*d_model)
        x = self.transformer1(x)  # (B*L//12, K, 4*d_model)
        x = x.reshape(B, L//12, K, -1).permute(0,2,1,3)  # (B, K, L/12, 4*d_model)

        x = x + skip2
        
        # --- 上采样路径 ---
        # 第二级上采样
        x = self.up_proj2(x)  # (B, K, L/12, 8*d_model)
        x = x.reshape(B, K, L//3, 2*self.d_model)  # (B, K, L/3, 2*d_model)
        
        # 第二级Transformer
        # x = x.reshape(B*K, L//3, -1)  # (B*K, L/3, 2*d_model)
        # x = self.transformer2(x)  # (B*K, L/3, 2*d_model)
        # x = x.reshape(B, K, L//3, -1)  # (B, K, L/3, 2*d_model)

        x = x.permute(0,2,1,3).reshape(B*L//3,K,-1) # (B*L//3, K, 4*d_model)
        x = self.transformer2(x)  # (B*L//3, K, 4*d_model)
        x = x.reshape(B, L//3, K, -1).permute(0,2,1,3)  # (B, K, L/3, 4*d_model)

        x = x + skip1  # 残差连接
        
        # 第一级上采样
        x = self.up_proj1(x)  # (B, K, L/3, 3*d_model)
        x = x.reshape(B, K, L, self.d_model)  # (B, K, L, d_model)
        x = x + skip0  # 残差连接
        
        # --- 双路径Transformer ---
        # 时间路径Transformer
        x = x.reshape(B*K, L, self.d_model)  # (B*K, L, d_model)
        x = self.transformer3(x)  # (B*K, L, d_model)
        x = x.reshape(B, K, L, self.d_model)  # (B, K, L, d_model)
        
        # 特征路径Transformer
        x = x.permute(0, 2, 1, 3)  # (B, L, K, d_model)
        x = x.reshape(B*L, K, self.d_model)  # (B*L, K, d_model)
        x = self.transformer4(x)  # (B*L, K, d_model)
        x = x.reshape(B, L, K, self.d_model)  # (B, L, K, d_model)
        x = x.permute(0, 2, 1, 3)  # (B, K, L, d_model)
        x = x + residual 

        x = self.output_proj_1(x)  # (B, K, L, d_model)
        x = self.output_proj_2(x)  # (B, K, L, 1)
        x = x.squeeze(-1)  # (B, K, L)
        
        # 最终残差连接
        final = x + ori
        return final


    
    
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