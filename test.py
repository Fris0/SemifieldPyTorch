import torch
from semifield.ops import SemiConv2d

conv = SemiConv2d(in_channels=3, out_channels=1, semifield_type="MaxMin", kernel_size=2, stride=1)
input = torch.zeros(2, 3, 10, 10)
input[:,:,5,5] = 1
output = conv.forward(input)