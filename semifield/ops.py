"""
Author: Mark Jansen
ops.py;

Folder that wraps the C++ or CUDA Semifield Convolution functions. This will allow
the C++ or CUDA implementation to work with the autograd system.

Check __init__.py for the actual explicit exports of this folder.
Meaning, which functions or classes are callable from the package semifield.
"""

import torch
import math
from semifield import conv2d


class MaxMin(torch.autograd.Function):
    """
    Create an autograd supported function.

    Define forward and backward to be the
    compiled semifield.dilation operations.
    """    
    @staticmethod
    def forward(ctx, input, kernel, padding):
        # Unpack padding values
        top, bottom = padding

        # Make input and kernel contiguous in memory for CUDA.
        input_contig = input.contiguous()
        kernel_contig = kernel.contiguous()

        # Do forward and store output for consequent steps and indicees for backward
        output, indicees = conv2d.forward(input_contig,
                                          kernel_contig,
                                          top,
                                          bottom
                                          )

        # Make indicees contiguous
        indicees_contig = indicees.contiguous()

        # Save indicees of X and Kernel where max was found
        ctx.save_for_backward(input_contig, kernel_contig, indicees_contig)
        ctx.padding(top, bottom)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Make grad_output from PyTorch contiguous
        grad_output_contig = grad_output.contiguous()

        # Retrieve saved tensors for backward
        input_contig, kernel_contig, indicees_contig = ctx.saved_tensors
        top, bottom = ctx.padding

        # Calculate the gradients with respect to the input and kernel
        grad_input, grad_kernel = conv2d.backward(grad_output_contig,
                                                  input_contig,
                                                  kernel_contig,
                                                  indicees_contig,
                                                  top,
                                                  bottom)
        
        # Return the grad outputs. Pytorch will update self.kernel of SemiConv2d.
        return grad_input, grad_kernel

# Dictionary contiang key word to class for SemiConv2d
SemiConv2dOptions = {"MaxMin": MaxMin}

class SemiConv2d(torch.nn.Module):
    """
    Class of Semifield Convolutions

    Sets up the initial values of the semifield conovolutions

    Supports forward and backward propagation with update of
    kernel values through gradient descent.

    First initialize the SemiConv2D, then use class functions.
    """

    def __init__(self, semifield_type, kernel_size=(3,3)):
        super().__init__()
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_size,
                                                     device='cuda',
                                                     requires_grad=True))
        self.semifield_type = semifield_type
        self.padding = self.calculate_padding()

    def forward(self, input):
        match self.semifield_type:
            case "MaxMin":
                return MaxMin.apply(input, self.kernel, self.padding)

    def calculate_padding(self):
        """
        Calculate the padding for symmetric and assymetric kernels
        required for same sized outputs.
        """

        # Obtain kernel height
        kernel_w, kernel_h = self.kernel.shape

        # If both sides are not even, throw an exception
        if kernel_w != kernel_h:
            raise Exception(ValueError)

        # Calculate total padding on height
        p_h = kernel_h - 1

        top = math.floor(p_h / 2)  # = left
        bottom = p_h - top  # = right

        return (top, bottom)  # Therefore, only need top and bottom
