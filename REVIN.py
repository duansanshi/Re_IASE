import torch
import torch.nn as nn


#期望输入(B,L,K)
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    def print_info(self):
        print("标准差:")
        print(self.stdev)
        print("均值:")
        print(self.mean)
        if self.affine:
            print("仿射权重:")
            print(self.affine_weight)
            print("仿射bias:")
            print(self.affine_bias)
    
class RevIN_REF(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN_REF, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x , ref,mode:str):
        if mode == 'norm':
            self._get_statistics(ref)
            x = self._normalize(x)
        elif mode == 'denorm':
            self._get_statistics(ref)
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    def print_info(self):
        print("标准差:")
        print(self.stdev)
        print("均值:")
        print(self.mean)
        if self.affine:
            print("仿射权重:")
            print(self.affine_weight)
            print("仿射bias:")
            print(self.affine_bias)

if __name__=="__main__":
    x=torch.tensor([[[ 0.4236,  0.0768, -0.2587,  1.2369],
         [ 0.0221,  0.5578,  0.6587, -0.8162],
         [-0.0695,  0.0101,  1.0673, -0.2255]],

        [[-0.2585,  0.5326,  0.7839,  0.7401],
         [ 1.3527,  0.8849, -1.0658,  0.3765],
         [ 0.3316,  0.0324, -0.4638,  0.0360]]])
    revin_layer = RevIN(num_features=4)
    print(x.shape)
    print(revin_layer(x,mode='norm'))
    print(x)