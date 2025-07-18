import torch
from semifield.ops import SemiConv2d

# Simple deterministic input
input = torch.zeros(3, 3, 3, dtype=torch.float32).cuda()
input[:, 1, 1] = 1.0  # Center pixel
input.requires_grad = True


print("INPUT:", input)

# Fixed kernel with known dtype and device
conv = SemiConv2d(in_channels=3, out_channels=3, semifield_type="SmoothMax", kernel_size=3, stride=1, groups=3, padding_mode="same", alpha=5)

# Run forward
output = conv(input)
print("Output:\n", output)

# Backward: simple loss
loss = output.sum()
#print(loss)
loss.backward()


print("Grad input:\n", input.grad)
print("Grad kernel:\n", conv.kernel.grad)