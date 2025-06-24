"""
Author: Mark Jansen
ops.py;

Folder that wraps the C++ or CUDA Semifield Convolution functions. This will allow
the C++ or CUDA implementation to work with the autograd system.

Check __init__.py for the actual explicit exports of this folder.
Meaning, which functions or classes are callable from the package semifield.
"""

import torch
import torch.nn.functional as F

import math

from semifield import conv2d


class MaxMin(torch.autograd.Function):
    """
    Create an autograd supported function.

    Define forward and backward to be the
    compiled semifield.dilation operations.

    inputs:
    in_channels  = channels of input
    out_channels = desired output channels
    input   = the input following the above shapes
    kernel  = the weight kernels equal to out channels
    padding = the padding required for correct convolutions
    stride  = the spacing between each convolution

    Output:
    The forward pass output values after convolution.
    
    Side-effects:
    Call the cuda conv2d forward function and update the values within
    the indicees and output contiguous tensor.
    """    
    @staticmethod
    def forward(ctx, in_channels, out_channels, input, kernel, padding, stride):
        # Unpack padding values
        left, right, top, bottom = padding
        pad_w = left + right
        pad_h = top  + bottom

        # Make input and kernel contiguous in memory for CUDA.
        input_contig = input.contiguous()
        kernel_contig = kernel.contiguous()

        # Do forward and store output for consequent steps and indicees for backward
        output, indicees = conv2d.max_min_forward(
                                                in_channels,
                                                out_channels,
                                                input_contig,
                                                kernel_contig,
                                                pad_w,
                                                pad_h,
                                                stride,
                                                )

        # Make indicees contiguous
        indicees_contig = indicees.contiguous()

        # Save indicees of X and Kernel where max was found
        ctx.save_for_backward(input_contig, kernel_contig, indicees_contig)
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.stride = stride

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Make grad_output from PyTorch contiguous
        grad_output_contig = grad_output.contiguous()

        # Retrieve saved tensors and params for backward
        input_contig, kernel_contig, indicees_contig = ctx.saved_tensors
        in_channels = ctx.in_channels
        out_channels = ctx.out_channels
        pad_w = ctx.w
        pad_h = ctx.h
        stride = ctx.stride

        # Calculate the gradients with respect to the input and kernel
        grad_input, grad_kernel = conv2d.max_min_backward(
                                                        in_channels,
                                                        out_channels,
                                                        grad_output_contig,
                                                        input_contig,
                                                        kernel_contig,
                                                        indicees_contig,
                                                        pad_w,
                                                        pad_h,
                                                        stride)
        
        # Return the grad outputs. Pytorch will update self.kernel of SemiConv2d.
        return  None, None, grad_input, grad_kernel, None, None  # Return size has to be equal to input size of forward

# Dictionary contiang key word to class for SemiConv2d
SemiConv2dOptions = {"MaxMin": MaxMin}

class SemiConv2d(torch.nn.Module):
    """
    Class of Semifield Convolutions

    Sets up the initial values of the semifield conovolutions

    Supports forward and backward propagation with update of
    kernel values through gradient descent.

    First initialize the SemiConv2D, then use class functions.

    inputs:
    N = Batch size
    in_channels  = channels of input
    out_channels = desired output channels
    semifield_type = MaxMin or MinPlus
    kernel_size  = Tuple of shape of the kernel (W, H)
    stride  = the spacing between each convolution

    Side-effects:
    Calls the MaxMin or MinPlus semifield convolution wrappers.
    """

    def __init__(self, in_channels, out_channels, semifield_type, kernel_size=3, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.kernel = None  # Lazy init
        self.padding = None
        self.kernel_size = kernel_size

        self.semifield_type = semifield_type

    def forward(self, input):
        # Ensure right dimension
        input = self.unsqueeze_4d(input)

        # Creat kernel with same dtype as input for proper cuda-kernel functioning.
        if self.kernel == None:
            self.kernel = torch.nn.Parameter(torch.zeros(
                                                    self.out_channels,
                                                    self.kernel_size,
                                                    self.kernel_size,
                                                    device='cuda',
                                                    dtype=input.dtype,
                                                    requires_grad=True))

        # After creation of kernel find padding required for input
        self.padding = self.calculate_padding()

        # Call correct semifield convolution
        match self.semifield_type:
            case "MaxMin":
                # Pad input then pass it including kernel and padding
                input = F.pad(input, pad=self.padding, mode='constant', value=float('-inf'))
                return MaxMin.apply(
                                    self.in_channels,
                                    self.out_channels,
                                    input,
                                    self.kernel,
                                    self.padding,
                                    self.stride,
                )
            case _:
                raise ValueError(f"Expected MaxMin or MinPlus but got {self.semifield}.")

    def unsqueeze_4d(self, input):
        """
        Extends dimension of input tensor such that
        it always holds proper dimensions. This
        reduces conditional checks in code.

        Output:
        Tensor with 4D shape
        """
        if input.dim() == 2:
            self.initial_dim = 2
            return input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 3:
            self.initial_dim = 3
            return input.unsqueeze(0)
        elif input.dim() == 4:
            self.initial_dim = 4
            return input
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, but got {input.dim()}D.")

    def calculate_padding(self):
        """
        Calculate the padding for symmetric and assymetric kernels
        required for same sized outputs.
        """
        # Calculate total padding on height
        _, _, H = self.kernel.size()
        p_h = H - 1

        top = left = math.floor(p_h / 2)
        bottom = right = p_h - top

        return (left, right, top, bottom)
