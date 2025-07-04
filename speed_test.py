import torch
import torch.nn.functional as F
from semifield.ops import SemiConv2d
import time

torch.cuda.empty_cache()
# Set seed for reproducibility
torch.manual_seed(0)

# Device
device = "cuda"

# Config
batch_size = 1000
channels = 30
height = width = 32
kernel_size = 3
stride = 2

# Input
input = torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)

# --- SemiConv2d MaxMin in "maxpool mode" ---
semi = SemiConv2d(
    in_channels=channels,
    out_channels=channels,
    semifield_type="MaxMin",
    kernel_size=kernel_size,
    stride=stride,
    groups=channels  # MaxPool behavior
)

semi_avg = 0
max_avg = 0
for i in range(1000):  
    # Time SemiConv
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out_semi = semi(input)
    loss_semi = out_semi.sum()
    loss_semi.backward()
    end.record()
    torch.cuda.synchronize()
    semi_time = start.elapsed_time(end)  # ms

    # --- PyTorch MaxPool ---
    input2 = input.detach().clone().requires_grad_()
    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    start.record()
    out_pool = pool(input2)
    loss_pool = out_pool.sum()
    loss_pool.backward()
    end.record()
    torch.cuda.synchronize()
    pool_time = start.elapsed_time(end)

    max_avg += pool_time
    semi_avg += semi_time

    # --- Compare results ---
print(f"⏱️  SemiConv2d Time: {semi_avg / 1000:.3f} ms")
print(f"⏱️  MaxPool2d Time: {max_avg / 1000:.3f} ms")

# ⏱️  SemiConv2d Time: 3.245 ms
# ⏱️  MaxPool2d Time: 1.084 ms