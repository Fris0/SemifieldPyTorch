import torch
from semifield import dilation

class Dilation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return dilation.forward(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a, grad_b = dilation.backward(grad_output, a, b)
        return grad_a, grad_b


def dilation_op(a, b):
    return Dilation.apply(a, b)