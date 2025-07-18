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
batch_size = 10000
channels = 10
height = width = 32
kernel_size = 5
stride = 2

# --- SemiConv2d MaxMin in "maxpool mode" ---
semi = SemiConv2d(
    in_channels=channels,
    out_channels=channels,
    semifield_type="MaxPlus",
    kernel_size=kernel_size,
    stride=stride,
    groups=channels,  # MaxPool behavior
    alpha=4,
    padding_mode="same"
)

semi_avg = 0
max_avg = 0
for i in range(1000):
    input = torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)

    # Time SemiConv
    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # time maxpool
    start.record()
    out_pool = pool(input)
    loss_pool = out_pool.sum()
    loss_pool.backward()
    end.record()
    torch.cuda.synchronize()
    pool_time = start.elapsed_time(end)

    #Semifield
    input2 = input.detach().clone().requires_grad_()
    start.record()
    out_semi = semi(input2)
    loss_semi = out_semi.sum()
    loss_semi.backward()
    end.record()
    torch.cuda.synchronize()
    semi_time = start.elapsed_time(end)  # ms

    max_avg += pool_time
    semi_avg += semi_time
    # --- Compare results ---
print(f"⏱️  SemiConv2d Time: {semi_avg / 1000:.3f} ms")
print(f"⏱️  MaxPool2d Time: {max_avg / 1000:.3f} ms")

#⏱️  SemiConv2d Time: 2.869 ms
#⏱️  MaxPool2d Time: 1.083 ms