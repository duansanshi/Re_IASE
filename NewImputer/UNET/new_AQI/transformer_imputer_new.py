import torch.nn as nn
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

######################
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGating(nn.Module):
    def __init__(self, num_layers=3, feature_dim=36):
        super().__init__()
        # 使用 1x1 卷积 (Conv1d) 在时间步维度上滑动计算权重
        # 输入通道为所有层的特征拼接 (num_layers * feature_dim)
        # 输出通道为层数 (num_layers)，即为每一层生成一个权重分数
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=num_layers * feature_dim, out_channels=feature_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim * 2, out_channels=num_layers, kernel_size=1)
        )

    def forward(self, skip_results):
        """
        参数:
        skip_results: 包含多个 tensor 的列表，每个 tensor 的形状为 [B, K, L]
        
        返回:
        final_pred: 融合后的输出 [B, K, L]
        attention_weights: 各层的注意力权重，可用于可视化分析 [B, num_layers, L]
        """
        # 1. 在特征维度 (dim=1) 拼接多层特征 -> [B, 3*K, L]
        stacked_features = torch.cat(skip_results, dim=1) 
        
        # 2. 计算注意力分数 (Logits) -> [B, 3, L]
        attention_scores = self.gate(stacked_features)
        
        # 3. 在层维度（dim=1）上做 Softmax，确保每个时间步的三层权重和为 1
        attention_weights = F.softmax(attention_scores, dim=1) # [B, 3, L]
        
        # 4. 加权融合
        # 将 skip_results 堆叠为 [B, 3, K, L]
        all_skips = torch.stack(skip_results, dim=1) 
        
        # 将权重扩展为 [B, 3, 1, L] 以便利用 PyTorch 的广播机制与 all_skips 相乘
        attention_weights_expanded = attention_weights.unsqueeze(2) 
        
        # 逐元素相乘后，在层维度 (dim=1) 进行求和，输出维度恢复为 [B, K, L]
        final_pred = torch.sum(all_skips * attention_weights_expanded, dim=1)
        
        return final_pred, attention_weights


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

        # Transformer编码器
        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=4*d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1
        )

        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2*d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1
        )
     
        self.transformer3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1
        )
        self.transformer4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1
        )
        
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
 
        
    def forward(self,x):
        """
        feature (B,K,L,d_model)-->the same size
        """
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
        return x + residual 



class transformer_imputer(nn.Module):
    def __init__(self, device, d_model=64, seq_len=36, feature_dim=36, pe_dim=128):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # 位置编码模块
        self.pe_dim = pe_dim
        self.pe_proj = nn.Linear(pe_dim+16, d_model)
        self.feature_pe = nn.Parameter(torch.zeros(1, feature_dim, 16))
        
        # 预计算序列位置编码
        positions = torch.arange(seq_len).float().to(device)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, pe_dim, 2).float() / pe_dim).to(device)
        pe = torch.zeros(seq_len, pe_dim).to(device)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
        self.register_buffer("seq_pe", pe)  # (seq_len, pe_dim)
      
        
        # 输入投影层
        self.input_proj = nn.Linear(2, d_model)  
        # 输出层
        self.output_proj_1 = nn.Linear(d_model, d_model)
        self.output_proj_2 = nn.Linear(d_model, 1)  
        # self.output_proj_11 = nn.Linear(d_model, d_model)
        # self.output_proj_12 = nn.Linear(d_model, 1)   
        # self.output_proj_21 = nn.Linear(d_model, d_model)
        # self.output_proj_22 = nn.Linear(d_model, 1)  
        # self.output_proj_31 = nn.Linear(d_model, d_model)
        # self.output_proj_32 = nn.Linear(d_model, 1)  
        self.to(device)

        self.UnetBlock1 = UnetBlock(self.device,self.d_model)
        self.UnetBlock2 = UnetBlock(self.device,self.d_model)
        self.UnetBlock3 = UnetBlock(self.device,self.d_model)
        #self.UnetBlock4 = UnetBlock(self.device,self.d_model)
        #self.fusion_weights = nn.Parameter(torch.ones(3))
        self.dynamic_gating = DynamicGating(num_layers=3, feature_dim=seq_len)

    def forward(self, x, cond_mask, tod=None, dow=None):
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
           
        ], dim=-1)  # (B, K, L, 2)
        
        # 输入投影
       
        x = self.input_proj(x)  # (B, K, L, d_model)
  
        # 添加位置编码

        seq_pe = self.seq_pe.unsqueeze(0).unsqueeze(0).expand(B,K,-1,-1)  # (1, 1, L, pe_dim)
        feature_pe = self.feature_pe.unsqueeze(2).expand(B,-1,L,-1) 
  
        pe = torch.cat([seq_pe, feature_pe], dim=-1)  # (B, K, L, pe_dim + feature_dim)
        pe = self.pe_proj(pe)  # (B, K, L, d_model)
        x = x + pe  # (B, K, L, d_model)


        
        # ----------------------------------------------------
        # 初始化用于深度监督的结果列表
        skip_results = [] 
        # ----------------------------------------------------
        
        # 1. Block 1
        x = self.UnetBlock1(x)
        x = x + pe
        
        # --- 深度监督投影和收集 (Block 1) ---
        # 投影：(B, K, L, d_model) -> (B, K, L, 1) -> (B, K, L)
        # 注意：这里的投影层是复用最后的 output_proj_1/2
        layer_pred = self.output_proj_1(x) 
        layer_pred = F.relu(layer_pred)
        layer_pred = self.output_proj_2(layer_pred).squeeze(-1) # (B, K, L)
        #layer_pred = layer_pred + ori  # 残差连接
        skip_results.append(layer_pred)
        # --------------------------------------
        
        # 2. Block 2
        x = self.UnetBlock2(x)
        x = x + pe
        
        # --- 深度监督投影和收集 (Block 2) ---
        layer_pred = self.output_proj_1(x) 
        layer_pred = F.relu(layer_pred)
        layer_pred = self.output_proj_2(layer_pred).squeeze(-1) # (B, K, L)
        #layer_pred = layer_pred + ori  # 残差连接
        skip_results.append(layer_pred)
        # --------------------------------------
        
        # 3. Block 3
        x = self.UnetBlock3(x)
        x = x + pe
        
        # --- 深度监督投影和收集 (Block 3) ---
        layer_pred = self.output_proj_1(x) 
        layer_pred = F.relu(layer_pred)
        layer_pred = self.output_proj_2(layer_pred).squeeze(-1) # (B, K, L)
        #layer_pred = layer_pred + ori  # 残差连接
        skip_results.append(layer_pred)
        # --------------------------------------
        # # 4. Block 4
        # x = self.UnetBlock4(x)
        # x = x + pe
        
        # # --- 深度监督投影和收集 (Block 4) ---
        # layer_pred = self.output_proj_1(x) 
        # layer_pred = F.relu(layer_pred)
        # layer_pred = self.output_proj_2(layer_pred).squeeze(-1) # (B, K, L)
        # skip_results.append(layer_pred)
        
        # ----------------------------------------------------
        # 最终输出计算 (主预测)
        # ----------------------------------------------------
        
        # 最终的特征融合（模仿您参考代码中的 skip_list 求和平均）
        # 这里的 final_x 是所有中间层预测结果的平均值
        #final_x = torch.sum(torch.stack(skip_results), dim=0) / len(skip_results)
       
        #final_x = 0.1 * skip_results[0] +0.2* skip_results[1] + 0.3 * skip_results[2] + 0.4 * skip_results[3]
        # w = F.softmax(self.fusion_weights, dim=0)
        # final_x = w[0]*skip_results[0] + w[1]*skip_results[1] + w[2]*skip_results[2]
        # (注意：如果您想保持原有的"特征经过所有Block后才投影"的逻辑，请使用下一段代码)
        
        #final_x, attention_weights = self.dynamic_gating(skip_results)  # (B, K, L)
        final_x = skip_results[-1]  # 直接使用最后一层的预测结果作为最终输出
        #final_x = 0.2 * skip_results[0] +0.3* skip_results[1] + 0.5 * skip_results[2] 
        
        # 最终残差连接
        #final_pred = final_x + ori
        final_pred = final_x 
        
        # 返回最终预测值 和 中间层预测列表
        return final_pred, skip_results

    
    
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
        #freq_coeffs = batch["freq"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)


        if True:
            coeffs = coeffs.permute(0, 2, 1)
        #freq_coeffs = freq_coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            #freq_coeffs,
        )