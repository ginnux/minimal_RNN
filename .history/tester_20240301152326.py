import torch
from model import MinimalRNNCell

# (Num, Length, inputsize)
a = torch.randn((64, 48, 1484))
net = MinimalRNNCell(1484, 8, 1484)
out = net(a, None)
print(out.shape)