import sys
sys.path.append("/home/duanlei/PriSTI")
from layers import *


class Guide_diff(nn.Module):
    def __init__(self, config, inputdim=1, target_dim=36, is_itp=False):
        super().__init__()
        self.channels = config["channels"]
        self.is_itp = is_itp
        self.itp_channels = None
       
        self.itp_channels = config["channels"]
        self.itp_projection = Conv1d_with_init(1, self.itp_channels, 1)

        self.itp_modeling = GuidanceConstruct(channels=self.itp_channels, nheads=config["nheads"], target_dim=target_dim,
                                        order=2, include_self=True, device=config["device"], is_adp=config["is_adp"],
                                        adj_file=config["adj_file"], proj_t=config["proj_t"])
        self.cond_projection = Conv1d_with_init(config["side_dim"], self.itp_channels, 1)
        self.itp_projection2 = Conv1d_with_init(self.itp_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        if config["adj_file"] == 'AQI36':
            self.adj = get_adj_AQI36()
        elif config["adj_file"] == 'metr-la':
            self.adj = get_similarity_metrla(thr=0.1)
        elif config["adj_file"] == 'pems-bay':
            self.adj = get_similarity_pemsbay(thr=0.1)
        elif config["adj_file"] == 'pems-08':
            self.adj = get_similarity_pems08(thr=0.1)
        elif config["adj_file"] == 'pems-04':
            self.adj = get_similarity_pems04(thr=0.1)
        self.device = config["device"]
        self.support = compute_support_gwn(self.adj, device=config["device"])
        self.is_adp = config["is_adp"]
        if self.is_adp:
            node_num = self.adj.shape[0]
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
            self.support.append([self.nodevec1, self.nodevec2])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        if self.config['model']['use_guide']:
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

    # def forward(self, x, side_info, diffusion_step, itp_x, cond_mask):
    #     if self.is_itp:
    #         x = torch.cat([x, itp_x], dim=1)
    #     B, inputdim, K, L = x.shape

    #     x = x.reshape(B, inputdim, K * L)
    #     x = self.input_projection(x)
    #     x = F.relu(x)
    #     x = x.reshape(B, self.channels, K, L)

    #     if self.is_itp:
    #         itp_x = itp_x.reshape(B, inputdim-1, K * L)
    #         itp_x = self.itp_projection(itp_x)
    #         itp_cond_info = side_info.reshape(B, -1, K * L)
    #         itp_cond_info = self.cond_projection(itp_cond_info)
    #         itp_x = itp_x + itp_cond_info
    #         itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
    #         itp_x = F.relu(itp_x)
    #         itp_x = itp_x.reshape(B, self.itp_channels, K, L)

    #     diffusion_emb = self.diffusion_embedding(diffusion_step)

    #     skip = []
    #     for i in range(len(self.residual_layers)):
    #         x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
    #         skip.append(skip_connection)

    #     x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
    #     x = x.reshape(B, self.channels, K * L)
    #     x = self.output_projection1(x)  # (B,channel,K*L)
    #     x = F.relu(x)
    #     x = self.output_projection2(x)  # (B,1,K*L)
    #     x = x.reshape(B, K, L)
    #     return x
    def forward(self, x, diffusion_step, side_info, itp_x, cond_mask): 
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

    
        itp_x = itp_x.reshape(B, 1, K * L)
        itp_x = self.itp_projection(itp_x)
        itp_cond_info = side_info.reshape(B, -1, K * L)
        itp_cond_info = self.cond_projection(itp_cond_info)
        itp_x = itp_x + itp_cond_info
        itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
        itp_x = F.relu(itp_x)
        itp_x = itp_x.reshape(B, self.itp_channels, K, L)
        
        # ... diffusion_emb 计算保持不变 ...
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip_results = [] # 用于存储每一层的处理后的结果
        
        for i in range(len(self.residual_layers)):
            # 每一层计算
            x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
            
            # --- [关键修改开始] ---
            # 我们拿到当前的 skip_connection，直接过一遍输出层
            # 注意：skip_connection 的形状是 (B, channels, K*L)
            
            # 1. 复用 output_projection1
            skip_connection = skip_connection.reshape(B, self.channels, K * L)
            layer_pred = self.output_projection1(skip_connection) 
            layer_pred = F.relu(layer_pred)
            
            # 2. 复用 output_projection2
            layer_pred = self.output_projection2(layer_pred) # 变成 (B, 1, K*L)
            
            # 3. 变形回 (B, K, L)
            layer_pred = layer_pred.reshape(B, K, L)
            
            skip_results.append(layer_pred)
            # --- [关键修改结束] ---

        # 原有的逻辑：计算最终的融合输出
        # 这里你可以选择保留原有的 sum 逻辑作为"主输出"
        # 或者直接把最后一层的 skip_results[-1] 当作主输出
        
        # 原有逻辑保留，作为最终 ensemble 结果
        # final_x = torch.sum(torch.stack([s for s in skip_results]), dim=0) / math.sqrt(len(self.residual_layers))
        #final_x = torch.sum(torch.stack([s for s in skip_results]), dim=0) / len(self.residual_layers)
        final_x = 0.1 * skip_results[0] +0.2* skip_results[1] + 0.3 * skip_results[2] + 0.4 * skip_results[3]  
        # 注意：因为上面的循环里我已经投影过了，这里逻辑稍微变一下：
        # 如果你想对齐的是“最终预测值”，上面的 skip_results 已经是预测值了，直接平均即可
        # 如果你坚持用原有逻辑对 skip connection 求和再投影，保持原有代码，
        # 但我们需要返回 skip_results 给 Loss 函数使用。
        
        # 修正建议：为了方便计算 Loss，返回两个值：
        # 1. final_pred: 模型最终的预测
        # 2. layer_preds: 中间层的预测列表
        
        # 重新计算原始逻辑的 final output (为了保持一致性)
        # 这里的实现取决于你是想让"每层预测的平均"作为结果，还是"特征和的预测"作为结果。
        # 假设保持原代码逻辑不变（特征和）：
        # (你需要重新收集原始的 skip tensor，或者仅仅返回 list 让外部计算 loss)
        
        return final_x, skip_results
    
    


class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, adj_file=None, is_cross_t=False, is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = SpatialLearning(channels=channels, nheads=nheads, target_dim=target_dim,
                                               order=order, include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file, proj_t=proj_t, is_cross=is_cross_s)

    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb
        
        y = self.forward_time(y, base_shape, itp_info)
        y = self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip

