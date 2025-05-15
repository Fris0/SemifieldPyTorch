"""
Author: Mark Jansen
ops.py;

Folder that wraps the C++ or CUDA functions. This will allow
the C++ or CUDA implementation to work with the autograd system.

Check __init__.py for the actual explicit exports of this folder.
Meaning, which functions are callable from the package semifield.
"""

import torch
from semifield import dilation

"""
Create an autograd supported function.

Define forward and backward to be the
compiled semifield.dilation operations.
"""
class Dilation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return dilation.forward(a.contiguous(), b.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a, grad_b = dilation.backward(grad_output.contiguous(), a.contiguous(), b.contiguous())
        return grad_a, grad_b


def dilation_op(a, b):
    return Dilation.apply(a, b)