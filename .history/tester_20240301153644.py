import torch
from model import MinimalRNNCell

# (Num, Length, inputsize)
a = torch.randn((64, 48, 1484)).to('cuda')
net = MinimalRNNCell(1484, 8, 8)
out = net(a, None)
print(out[0].shape)
print(out[1].shape)