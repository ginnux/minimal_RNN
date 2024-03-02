import torch
from model import MinimalRNN

# (Num, Length, inputsize)
a = torch.randn((48, 1484)).to("cuda")
# (Num_layers, Hiddensize)
h = torch.randn((6, 8)).to("cuda")
net = MinimalRNN(1484, 8, 8, num_layers=6, device="cuda")
out = net(a, torch.ra)
print(out[0].shape)
print(out[1].shape)
