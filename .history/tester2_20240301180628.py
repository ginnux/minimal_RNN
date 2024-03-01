import torch
from model import MinimalRNN

# (Num, Length, inputsize)
a = torch.randn((64, 48, 1484)).to("cuda")
net = MinimalRNN(1484, 8, 8, 3)
out = net(a, None)
print(out[0].shape)
print(out[1].shape)
