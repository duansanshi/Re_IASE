import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe
from ScConv import SRU
import math

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

class AFTFull(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full

        Number of heads is 1 as done in the paper
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.wbias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen))
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x):
        x = x.transpose(1, 2)
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        temp_wbias = self.wbias[:T, :T].unsqueeze(0) # sequences can still be variable length

        '''
        From the paper
        '''
        Q_sig = torch.sigmoid(Q)
        kv = contract("cln,cln->cln", torch.exp(K), V)
        temp = contract("cln,cln->cln", torch.exp(temp_wbias), kv)
        weighted = temp / (torch.exp(temp_wbias) @ torch.exp(K))
        Yt = contract("cnl,cnl->cnl",Q_sig, weighted)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt, Yt

class ScaleDotProductAttention(nn.Module):
    def __init__(self, attn_dropout, temperature=1):
        super(ScaleDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.SRU = SRU(64,
                       group_num=4,
                       gate_treshold=0.5)
        self.gate_treshold1 = 0.6
        self.gate_treshold2 = 0.3
        self.sigomid = nn.Sigmoid()
        channels = 64
        self.h = 207
        self.kernel_dim = 24
        self.kernel = torch.randn(self.h, channels, self.kernel_dim).uniform_(0.0001,0.001).cuda(2)
        nn.init.xavier_uniform_(self.kernel)
        self.kernel.require_grad = True




    def forward(self, q, k, v, base_shape):

        bz, nodeq, d = q.shape

        bz2, _, _, _ = base_shape

        kernel1 = self.kernel.repeat(bz2, 1, 1)

        y_f = contract('cnl,cnl->cnl', q, k)


        attn = self.dropout(F.softmax(y_f, dim=-1))

        attn = attn * (1 / torch.sqrt(torch.tensor(d))) + kernel1
        output = contract('cnl,cnl->cnl', attn, v)


        return output, attn


'''core module'''
class spa(nn.Module):
    def __init__(self, attn_dropout, temperature=1):
        super(spa, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.SRU = SRU(64,
                       group_num=4,
                       gate_treshold=0.5)
        self.gate_treshold1 = 0.6
        self.gate_treshold2 = 0.3
        self.sigomid = nn.Sigmoid()
        channels = 64
        self.h = 24
        self.kernel_dim = 135
        self.kernel = torch.randn(self.h, channels, self.kernel_dim).uniform_(0.0001,0.001).cuda(2)
        nn.init.xavier_uniform_(self.kernel)
        self.kernel.require_grad = True




    def forward(self, q, k, v, base_shape):

        bz, nodeq, d = q.shape

        bz2, _, _, _ = base_shape

        kernel1 = self.kernel.repeat(bz2, 1, 1)

        y_f = contract('cnl,cnl->cnl', q, k)

        attn = self.dropout(F.softmax(y_f, dim=-1))

        attn = attn * (1 / torch.sqrt(torch.tensor(d))) + kernel1
        output = contract('cnl,cnl->cnl', attn, v)


        return output, attn



''' Score-CDM core module '''
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dimension, in_dimension, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.mid_dimension = dimension


        global nhead

        nhead = n_heads

        if (in_dimension % n_heads) == 0:

            self.w_q = nn.Linear(dimension, dimension * n_heads)
            self.w_k = nn.Linear(dimension, dimension * n_heads)
            self.w_v = nn.Linear(dimension, dimension * n_heads)



            self.proj_d = nn.Linear(dimension * n_heads, dimension)
            self.proj_att = nn.Linear(dimension * n_heads, in_dimension)
            self.attention = ScaleDotProductAttention(dropout)


        else:
            n_heads = 5
            self.w_q = nn.Linear(dimension, dimension * n_heads)
            self.w_k = nn.Linear(dimension, dimension * n_heads)
            self.w_v = nn.Linear(dimension, dimension * n_heads)

            self.proj_d = nn.Linear(dimension * n_heads, dimension)
            self.proj_att = nn.Linear(dimension * n_heads, in_dimension)
            '''Score-CDM,Core module'''
            self.attention = spa(dropout) #



        self.addD = nn.Sequential(
            nn.Linear(in_dimension, dimension),
            nn.Dropout(dropout),
            nn.ReLU(inplace=False),
        )

        L = 24
        a = torch.zeros(24)
        b = torch.linspace(0, 0, 24)
        for i in range(24):
            a1 = a[i] * torch.cos(torch.linspace(-1 * i * math.pi, i * math.pi, 24))
            b = b + a1

        self.c = nn.Parameter(b.clone().detach().cuda(2))

        self.subD = nn.Sequential(
                nn.Linear(dimension, in_dimension),
                nn.Dropout(dropout),
                nn.ReLU(inplace=False),
            )
        self.SRU = SRU(64,
                       group_num=4,
                       gate_treshold=0.5)



    def forward(self, q, k, v, base_shape):
        bz, nodeq, d = q.shape
        _, nodek, d = k.shape
        _, nodev, _ = v.shape
        residual = q
        B, channel, K, L = base_shape


        q_s = self.w_q(self.addD(q))
        k_s = self.w_k(self.addD(k))
        L = d
        k_f = torch.fft.rfft(self.c, n=2 * L)  # (C H L)

        u_f = torch.fft.rfft(v, n=2 * L)  # (B H L)

        y_f = contract('cnl,l->l', u_f, k_f)

        y = torch.fft.irfft(y_f, n=2 * 24)[..., :24]  # (B C H L)
        v = v + y
        v_s = self.w_v(self.addD(v))


        context, attn = self.attention(q_s, k_s, v_s, base_shape)
        context = context.contiguous().view(bz, nodeq, -1)

        output = self.subD(self.proj_d(context))


        return output + residual, attn

class FeedForwardNet(nn.Module):
    def __init__(self, dimension, dropout):
        super(FeedForwardNet, self).__init__()
        self.ff = nn.Sequential(
                nn.Linear(dimension, 16 * dimension),
                nn.ReLU(inplace=False),
                nn.Linear(16 * dimension, dimension),
                nn.Dropout(dropout),
                nn.ReLU(inplace=False),
            )

    def forward(self, inputs):
        # bz, region, channel
        residual = inputs
        output = self.ff(inputs)
        return residual + output

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, n_dimension, in_dimension, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(n_heads, n_dimension, in_dimension, dropout)
        self.ffn = FeedForwardNet(in_dimension, dropout)

    def forward(self, pre_d, d, adj = None):
        d = d.to(d.device)
        outputs = torch.tensor([], requires_grad = True)
        outputs, attn = self.attn(pre_d, d, d)
        outputs1 = outputs.clone()
        # outputs: (bsize, node, dimension) 
        outputs1 = self.ffn(outputs)
        return outputs1, attn

''' just make use of encoder layer of transformer for parameter decay '''
class ImpTransformer(nn.Module):
    def __init__(self, layers, n_heads, n_dimension, in_dimension, dropout=0.2):
        super(ImpTransformer, self).__init__()
        self.encoder = EncoderLayer(n_heads, n_dimension, in_dimension, dropout)
        self.ELayers = nn.ModuleList(
            [self.encoder for _ in range(layers)])

    def forward(self, pre_d, d, adj = None):
        enc_outputs = d
        for layer in self.ELayers:
            enc_outputs, attn1 = layer(pre_d, enc_outputs, adj)

        return enc_outputs, attn1


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

