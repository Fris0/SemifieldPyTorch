import torch
from semifield.ops import SemiConv2d

conv = SemiConv2d(in_channels=1, out_channels=2, semifield_type="MaxMin", kernel_size=2, stride=2)
input = torch.zeros(1, 1, 5, 5, dtype=torch.float32).cuda()
conv(input)