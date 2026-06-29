import torch

a = torch.randn(5,10)
b = torch.randn(10,5)
c = a@b
print(c.shape)
print(c)