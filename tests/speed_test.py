import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from semifield.ops import SemiConv2d

# Device
device = "cuda"
torch.manual_seed(0)
torch.cuda.empty_cache()

# Config
batch_size = 10240
channels = 10
height = width = 32
kernel_size = 3
stride = 2

# --- Setup modules ---
semi = SemiConv2d(
    in_channels=channels,
    out_channels=channels,
    semifield_type="MaxPlus",
    kernel_size=kernel_size,
    stride=stride,
    groups=channels,
    alpha=0,
    dtype=torch.float32
).to(device)

maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride).to(device)

# Shared input
input_tensor = torch.randn(batch_size, channels, height, width, device=device)

# --- Define wrapper functions ---
def run_semiconv():
    input_ = input_tensor.detach().clone().requires_grad_()
    out = semi(input_)
    loss = out.sum()
    loss.backward()

def run_maxpool():
    input_ = input_tensor.detach().clone().requires_grad_()
    out = maxpool(input_)
    loss = out.sum()
    loss.backward()

# --- Benchmark ---
t1 = benchmark.Timer(
    stmt="run_semiconv()",
    setup="from __main__ import run_semiconv",
    num_threads=1,
    label="SemiConv2d",
    sub_label="Forward+Backward+Update",
    description="MaxPlus"
)

t2 = benchmark.Timer(
    stmt="run_maxpool()",
    setup="from __main__ import run_maxpool",
    num_threads=1,
    label="MaxPool2d",
    sub_label="Forward+Backward",
    description="cuDNN"
)

results = benchmark.Compare([t1.timeit(40), t2.timeit(40)])
results.print()
