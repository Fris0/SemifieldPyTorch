import torch
from semifield.ops import SemiConv2d

# Create model
conv = SemiConv2d(in_channels=1, out_channels=1, semifield_type="MaxMin", kernel_size=3, stride=1)

# Dummy input: single 5x5 image with a 1 in the center
input = torch.zeros(1, 1, 5, 5, dtype=torch.float32).cuda()
input[:, :, 2, 2] = 1.0
input.requires_grad_()

# First forward to initialize kernel (if needed)
output = conv(input)

# Now create optimizer (after kernel exists)
optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)

# Save kernel before update
before = conv.kernel.detach().clone()

# Compute dummy loss and backward
loss = output.sum()
loss.backward()

# Apply optimizer step
optimizer.step()

# Save kernel after update
after = conv.kernel.detach().clone()

print("Kernel before step:\n", before)
print("Kernel after step:\n", after)

# ✅ Assert that kernel was updated
assert not torch.allclose(before, after), "Kernel was not updated!"
print("✅ Kernel update confirmed!")
