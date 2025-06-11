import torch
from semifield.ops import SemiConv2d

conv = SemiConv2d("MaxMin", 5, 5, kernel_size=(4,4))

