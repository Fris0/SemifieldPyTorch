import torch
from semifield import dilation_op

a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

out = dilation_op(a, b)
loss = out.sum()
loss.backward()

print(a.grad)  # Should print: tensor([4., 5., 6.])
print(b.grad)  # Should print: tensor([1., 2., 3.])