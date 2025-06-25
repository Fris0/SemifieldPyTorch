import torch
from semifield.ops import SemiConv2d

conv = SemiConv2d(in_channels=1, out_channels=2, semifield_type="MaxMin", kernel_size=3, stride=1)
input = torch.zeros(2, 1, 5, 5, dtype=torch.float32, requires_grad=True).cuda()
input[:,:, 2, 2] = 1
conv(input)