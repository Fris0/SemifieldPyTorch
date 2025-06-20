import torch
from semifield.ops import SemiConv2d

conv = SemiConv2d(in_channels=1, out_channels=1, semifield_type="MaxMin", kernel_size=3, stride=1)
input = torch.zeros(1, 1, 3, 3)
print(input)
input[:,:,1,1] = 1
output = conv.forward(input)