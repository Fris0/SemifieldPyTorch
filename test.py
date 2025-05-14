import torch
from semifield import dilation

a = torch.tensor([1.0, 2.0], requires_grad=True)
b = torch.tensor([3.0, 4.0], requires_grad=True)

out = dilation.forward(a, b)

# Some scalar loss (required for backward)
loss = out.sum()

loss.backward()

print(a.grad)  # Will show the gradient of loss w.r.t a
print(b.grad)  # Will show the gradient of loss w.r.t b
