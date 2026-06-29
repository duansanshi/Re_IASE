import torch

from layers import *
from transformer import *
from modeltcn import TCN

class Guide_diff(nn.Module):
    def __init__(self, config, inputdim=1, target_dim=36, is_itp=False):
        super().__init__()
        self.channels = config["channels"]
        self.is_itp = is_itp
        self.itp_channels = None
        if self.is_itp:
            self.itp_channels = config["channels"]
            self.itp_projection = Conv1d_with_init(inputdim-1, self.itp_channels, 1)

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
            self.adj = torch.randn(325,325).uniform_(0., 0.1)
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


    def forward(self, x, side_info, diffusion_step, itp_x, cond_mask):
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        if self.is_itp:
            itp_x = itp_x.reshape(B, inputdim-1, K * L)
            itp_x = self.itp_projection(itp_x)
            itp_cond_info = side_info.reshape(B, -1, K * L)
            itp_cond_info = self.cond_projection(itp_cond_info)
            itp_x = itp_x + itp_cond_info
            itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
            itp_x = F.relu(itp_x)
            itp_x = itp_x.reshape(B, self.itp_channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for i in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, adj_file=None, is_cross_t=False, is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = SpatialLearning(channels=channels, nheads=nheads, target_dim=target_dim,
                                               order=order, include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file, proj_t=proj_t, is_cross=is_cross_s)
        self.update_gate = MessagePN2(c_in=24, c_out=24, heads=nheads, layers=1, channels=channels)
        self.spa_gate = Messagespa(c_in=207, c_out=207, heads=nheads, layers=1, channels=channels)
        self.tcn = TCN(24, 24, [24,24,24])
        adj1 = torch.randn(325,325).to(device)
        adj1.require_grad = True
        self.impgconv = ImpSGConv(layers=1, adj1=adj1)

    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x

        y = self.forward_time(y, base_shape)
        # y = self.tcn(y)
        y = self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
        # y = self.forward_feature(y, base_shape)
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

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        C = channel
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        attn = torch.zeros_like(y)
        # y = y.reshape(B, channel, K, L).reshape(B * channel, K, L)
        # y = torch.cat([self.time_shift(y)[:, :L, :C // 2], y[:, :L, C // 2:]], dim=2).transpose(1, 2) # 只需增加这句

        y, attn = self.update_gate(y, base_shape)
        # y = self.tcn(y)
        # y = self.impgconv(y, base_shape)

        # y = y.reshape(B, channel, K, L).reshape(B, channel, K*L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)

        return y



class MessagePN2(nn.Module):
    def __init__(self, c_in, c_out, heads, layers, channels):
        super(MessagePN2, self).__init__()
        self.c_in = c_in
        self.c_tmp = 2 * c_in

        self.attn = MultiHeadAttention(heads, 3, self.c_in, 0.1)
        # self.linear = nn.Linear(self.c_in, 2 * self.c_in)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.c_in, 6 * (self.c_tmp + self.c_in)),  ###
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(6 * (self.c_tmp + self.c_in), c_out),
            nn.ReLU(inplace=False),
        )
        self.SRU = SRU(64,
                       group_num=4,
                       gate_treshold=0.5)

    def forward(self, data, base_shape):

        out, attn = self.attn(data, data, data, base_shape)
        out1 = torch.cat((out, data), dim=-1)


        return self.mlp(out1), attn




class Messagespa(nn.Module):
    def __init__(self, c_in, c_out, heads, layers, channels):
        super(Messagespa, self).__init__()
        self.c_in = c_in
        self.c_tmp = 2 * c_in

        self.attn = MultiHeadAttention(heads, 27, self.c_in, 0.1)
        # self.linear = nn.Linear(self.c_in, 2 * self.c_in)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.c_in, 2 * (self.c_tmp + self.c_in)),  ###
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(2 * (self.c_tmp + self.c_in), c_out),
            nn.ReLU(inplace=False),
        )
        self.SRU = SRU(64,
                       group_num=4,
                       gate_treshold=0.5)

    def forward(self, data, base_shape):
        # context = context.reshape(base_shape)
        # context = self.SRU(context)
        # context = context.reshape(B * channel, K, L)

        out, attn = self.attn(data, data, data, base_shape)
        out1 = torch.cat((out, data), dim=-1)
        # out = self.linear(data)

        return self.mlp(out1), attn





class ImpSGConv(nn.Module): #   (B H L)
    def __init__(self, layers, adj1 ,dropout=0.2):
        super(ImpSGConv, self).__init__()

        self.y = None
        u1, self.e, v = torch.linalg.svd(adj1)
        self.encoder = GConv(
            e = self.e,
            u1 = u1,
            v = v,
            d_model=64,
            adj=adj1,
            d_state=64,
            l_max=25,
            bidirectional=True,
            kernel_dim=32,
            n_scales=None,
            decay_min=2,
            decay_max=2,
        )
        self.ELayers = nn.ModuleList(
            [self.encoder for _ in range(1)])
    def forward(self, train, base_shape, return_kernel=True):
        y = train
        # train = train.reshape(base_shape)
        B, channel, K, L = base_shape
        # train = train.reshape(B * channel, K, L)
        k2 = None


        for layer in self.ELayers:

            y, k1 = layer(y, base_shape)

            y = (y + train) / 2

        y = y.reshape(base_shape)
        y = y.reshape(B, channel, K * L)
        return y
