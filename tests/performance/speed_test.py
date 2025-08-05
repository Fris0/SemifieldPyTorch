import torch
import pandas as pd
import numpy as np
import itertools
import math
import pynvml
from semifield import SemiConv2d


# Initialize NVML
pynvml.nvmlInit()

# Get handle to GPU 0 (change if using multi-GPU)
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Define column names
columns = ['semifield', 'kernel_size', 'padding','mb','stride','input_size','channels','mean_time_ms','std_dev_ms', 'clock']

# Create empty DataFrame with those columns
df = pd.DataFrame(columns=columns)

# Save to CSV with headers
print("Give GPU Name:")
file = f"./tests/performance/performance{input()}.csv"
#df.to_csv(file, index=False)

def run_with_backward(layer, x):
    out = layer(x)
    loss = out.sum()
    loss.backward()

# Device
device = "cuda"
torch.manual_seed(0)
torch.cuda.empty_cache()

# Config
target_mb = list(range(6, 78, 6))
batches = [
    (math.floor((mb * 1024 * 1024) / (10 * 64 * 64 * 4)), mb)
    for mb in target_mb
]
kernel_sizes = [2, 3, 5]
semifields = ['MaxPlus', 'MinPlus', 'SmoothMax']
in_out_channels = [(20, 20)]
image_size = [(128, 128)]
stride = [2]

# Padding config
kernel_padding = [(k, 0 if k == 2 else 'same') for k in kernel_sizes]

# All combinations
options = list(itertools.product(batches, kernel_padding, semifields, in_out_channels, image_size, stride))

#for idx, ((batch_size, mb), (k, pad), semifield, (in_channel, out_channel), (h, w), s) in enumerate(options):
#    try:
#        semi = SemiConv2d(
#            in_channels=in_channel,
#            out_channels=out_channel,
#            semifield_type=semifield,
#            kernel_size=k,
#            stride=s,
#            groups=in_channel,
#            padding_mode=pad,
#            dtype= torch.float32
#        ).to(device)
#
#        # Warm-up (1 forward + backward)
#        for _ in range(50):
#            run_with_backward(semi, torch.randn(100, in_channel, h, w, device=device, requires_grad=True))
#
#        # Benchmark 500 times (forward + backward)
#        # Get current SM clock speed (in MHz)
#        sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
#        times = []
#        for _ in range(1000):
#            input_tensor = torch.randn(batch_size, in_channel, h, w, device=device, requires_grad=True)
#            torch.cuda.synchronize()
#            start = torch.cuda.Event(enable_timing=True)
#            end = torch.cuda.Event(enable_timing=True)
#
#            start.record()
#            out = semi(input_tensor)
#            loss = out.sum()
#            loss.backward()
#            end.record()
#            torch.cuda.synchronize()
#
#            elapsed_ms = start.elapsed_time(end)
#            times.append(elapsed_ms)
#
#        mean_time = np.mean(times)
#        std_dev = np.std(times)
#
#        result = [{
#            'semifield': semifield,
#            'kernel_size': k,
#            'padding': pad,
#            'mb': mb,
#            'stride': s,
#            'input_size': f'{h}x{w}',
#            'channels': [in_channel, out_channel],
#            'mean_time_ms': mean_time,
#            'std_dev_ms': std_dev,
#            'clock': sm_clock
#        }]
#
#        df = pd.read_csv(file)
#        df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)
#        df.to_csv(file, index=False)
#
#
#    except RuntimeError as e:
#        if 'out of memory' in str(e):
#            print(f'Skipping OOM config: bs={batch_size}, size={h}x{w}, ch={in_channel}, sf={semifield}')
#            torch.cuda.empty_cache()
#            continue
#        else:
#            raise e
#
#    # Clean up
#    del input_tensor
#    del semi
#    torch.cuda.empty_cache()

def calculate_padding(k):
    """
    Calculate the padding for symmetric and assymetric kernels
    required for same sized outputs.
    Output: left, right top and bottom, where
    each variable represents the padding
    on that side of the input.
    """
    # Calculate total padding on height
    H = k
    p_h = H - 1
    top = left = math.floor(p_h / 2)
    bottom = right = p_h - top
    return (top, bottom)


# Config
kernel_sizes = [2, 3, 5]
kernel_padding = [(k, 0 if k == 2 else calculate_padding(k)) for k in kernel_sizes]
in_out_channels = [(20, 20)]
image_size = [(128, 128)]
stride = [2]

options = itertools.product(batches, kernel_padding, in_out_channels, image_size, stride)

for idx, ((batch_size, mb), (k, pad), (in_channel, out_channel), (h, w), s) in enumerate(options):
    try:
        maxpool = torch.nn.MaxPool2d(
            kernel_size=k,
            stride=s,
            padding=pad,
        ).to(device)

        # Warm-up (1 forward + backward)
        for _ in range(50):
            run_with_backward(maxpool, torch.randn(100, in_channel, h, w, device=device, requires_grad=True))

        # Benchmark 500 times (forward + backward)

        times = []
        sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        for _ in range(1000):
            input_tensor = torch.randn(batch_size, in_channel, h, w, device=device, requires_grad=True)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            out = maxpool(input_tensor)
            loss = out.sum()
            loss.backward()
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end)
            times.append(elapsed_ms)

        mean_time = np.mean(times)
        std_dev = np.std(times)

        result = [{
            'semifield': "Standard",
            'kernel_size': k,
            'padding': pad,
            'mb': mb,
            'stride': s,
            'input_size': f'{h}x{w}',
            'channels': [in_channel, out_channel],
            'mean_time_ms': mean_time,
            'std_dev_ms': std_dev,
            'clock': sm_clock
        }]

        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)
        df.to_csv(file, index=False)


    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'Skipping OOM config: bs={batch_size}, size={h}x{w}, ch={in_channel}, sf={semifield}')
            torch.cuda.empty_cache()
            continue
        else:
            raise e