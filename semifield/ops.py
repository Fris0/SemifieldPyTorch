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


# Helper functions to minimize re-used code.
def make_contiguous(one, two):
    """
    Obtain input and kernel and make them contiguous in memory.

    input: The input given by the user of type torch.tensor
    kernel: kernel defined in SemiConv2d.forward of type torch.tensor
    
    Output:
    Two contigious n-dimensional tensors.
    """
    return one.contiguous(), two.contiguous()

def save_ctx_tensors(ctx, input_contig, kernel_contig, input_indices, kernel_indices, in_channels, out_channels):
    """
    Store values required for backward in ctx class as attributes.

    Output: None
    Side-effects:
    Stores attributes inside ctx class.
    """
    ctx.save_for_backward(input_contig, kernel_contig, input_indices, kernel_indices)
    ctx.in_channels = in_channels
    ctx.out_channels = out_channels

class MaxPlus(torch.autograd.Function):
    """
    Create an autograd supported function.

    Define forward and backward to be the
    compiled conv2d operations.

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
    def forward(ctx, in_channels, out_channels, input, kernel, stride, groups):

        # Make input and kernel contiguous in memory for CUDA.
        input_contig, kernel_contig = make_contiguous(input, kernel)

        # Do forward and store output for consequent steps and indicees for backward
        output, input_indices, kernel_indices = conv2d.max_plus_forward(
                                                in_channels,
                                                out_channels,
                                                input_contig,
                                                kernel_contig,
                                                stride,
                                                groups
                                                )

        # Make indicees contiguous
        input_indices, kernel_indices = make_contiguous(input_indices, kernel_indices)

        # Save indicees of X and Kernel where max was found
        save_ctx_tensors(ctx, input_contig, kernel_contig,
                         input_indices, kernel_indices, in_channels,
                         out_channels)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Make grad_output from PyTorch contiguous
        grad_output_contig = grad_output.contiguous()

        # Retrieve saved tensors and params for backward
        input_contig, kernel_contig, input_indices, kernel_indices = ctx.saved_tensors
        in_channels = ctx.in_channels
        out_channels = ctx.out_channels

        # Calculate the gradients with respect to the input and kernel
        grad_input, grad_kernel = conv2d.max_plus_backward(
                                                        in_channels,
                                                        out_channels,
                                                        grad_output_contig,
                                                        input_contig,
                                                        kernel_contig,
                                                        input_indices,
                                                        kernel_indices)
        
        # Return the grad outputs. Pytorch will update self.kernel of SemiConv2d.
        return None, None, grad_input, grad_kernel, None, None # Return size has to be equal to input size of kernel

class MinPlus(torch.autograd.Function):
    """
    Create an autograd supported function.

    Define forward and backward to be the
    compiled conv2d operations.

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
    def forward(ctx, in_channels, out_channels, input, kernel, stride, groups):

        # Make input and kernel contiguous in memory for CUDA.
        input_contig, kernel_contig = make_contiguous(input, kernel)

        # Do forward and store output for consequent steps and indicees for backward
        output, input_indices, kernel_indices = conv2d.min_plus_forward(
                                                in_channels,
                                                out_channels,
                                                input_contig,
                                                kernel_contig,
                                                stride,
                                                groups
                                                )

        # Make indicees contiguous
        input_indices, kernel_indices = make_contiguous(input_indices, kernel_indices)

        # Save indicees of X and Kernel where max was found
        save_ctx_tensors(ctx, input_contig, kernel_contig,
                         input_indices, kernel_indices, in_channels,
                         out_channels)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Make grad_output from PyTorch contiguous
        grad_output_contig = grad_output.contiguous()

        # Retrieve saved tensors and params for backward
        input_contig, kernel_contig, input_indices, kernel_indices = ctx.saved_tensors
        in_channels = ctx.in_channels
        out_channels = ctx.out_channels

        # Calculate the gradients with respect to the input and kernel
        grad_input, grad_kernel = conv2d.min_plus_backward(
                                                        in_channels,
                                                        out_channels,
                                                        grad_output_contig,
                                                        input_contig,
                                                        kernel_contig,
                                                        input_indices,
                                                        kernel_indices)
        
        # Return the grad outputs. Pytorch will update self.kernel of SemiConv2d.
        return None, None, grad_input, grad_kernel, None, None  # Return size has to be equal to input size of kernel

class SmoothMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_channels, out_channels, input, kernel, stride, alpha, groups):

        # Make input and kernel contiguous in memory for CUDA.
        input_contig, kernel_contig = make_contiguous(input, kernel)

        # Do forward and store output for consequent steps and indicees for backward
        output = conv2d.smooth_max_forward(
                                            in_channels,
                                            out_channels,
                                            input_contig,
                                            kernel_contig,
                                            stride,
                                            alpha,
                                            groups
                                            )[0]

        # Save indicees of X and Kernel where max was found
        ctx.save_for_backward(input_contig, kernel_contig)
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.stride = stride
        ctx.alpha = alpha
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Make grad_output from PyTorch contiguous
        grad_output_contig = grad_output.contiguous()

        # Retrieve saved tensors and params for backward
        input_contig, kernel_contig = ctx.saved_tensors
        in_channels = ctx.in_channels
        out_channels = ctx.out_channels
        stride = ctx.stride
        alpha = ctx.alpha
        groups = ctx.groups

        # Calculate the gradients with respect to the input and kernel
        grad_input, grad_kernel = conv2d.smooth_max_backward(
                                                        in_channels,
                                                        out_channels,
                                                        grad_output_contig,
                                                        input_contig,
                                                        kernel_contig,
                                                        stride,
                                                        alpha,
                                                        groups)
        
        # Return the grad outputs. Pytorch will update self.kernel of SemiConv2d.
        return None, None, grad_input, grad_kernel, None, None, None # Return size has to be equal to input size of kernel

class SemiConv2d(torch.nn.Module):
    """
    Class of Semifield Convolutions

    Sets up the initial values of the semifield conovolutions

    Supports forward and backward propagation with update of
    kernel values through gradient descent.

    First initialize the SemiConv2D, then use class functions.

    inputs:
    in_channels  = channels of input
    out_channels = desired output channels
    semifield_type = MaxMin or MinPlus
    kernel_size  = Tuple of shape of the kernel (W, H)
    stride  = the spacing between each convolution
    groups = splits in_channels over groups
    alpha = the Smooth Max parameter
    groups = the amount of ways to split the input channels
    padding = Tuple (left, right, top, bottom)

    Side-effects:
    Calls the MaxMin or MinPlus semifield convolution wrappers.
    """

    def __init__(self, in_channels, out_channels, semifield_type, kernel_size=3, stride=1, alpha=1, groups=1, padding=(0,0,0,0), padding_mode=None):
        super().__init__()
        # Variables that define output shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Groups where groups equal in_channels causes pooling
        self.groups = groups

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups.")

        self.channels_per_group = in_channels // groups

        # Only used during SmoothMax semifield convolution
        self.alpha = float(alpha)

        if self.alpha < 0.0:
            raise ValueError("Alpha should be greater then 0")

        self.kernel = None  # Can be set before forward. Useful for structuring kernel functions.
        self.padding = padding # Lazy init
        self.padding_mode = padding_mode

        self.kernel_size = kernel_size

        # String used in case switch for correct smeifield convolution
        self.semifield_type = semifield_type

    def forward(self, input):
        # Ensure right dimension
        input = self.unsqueeze_4d(input)

        # Create kernel with same dtype as input for proper cuda-kernel functioning.
        if self.kernel == None:
            self.kernel = torch.nn.Parameter(torch.zeros(
                                            self.out_channels,
                                            self.channels_per_group,
                                            self.kernel_size,
                                            self.kernel_size,
                                            device=input.device,
                                            dtype=input.dtype
            ))
            self.register_parameter("kernel", self.kernel)

        # After creation of kernel find padding required for input
        if self.padding_mode == "same":
            self.padding = self.calculate_padding()

        # Call correct semifield convolution
        match self.semifield_type:
            case "MaxPlus":
                # Pad input then pass it including kernel and padding
                input = F.pad(input, pad=self.padding, mode='constant', value=float('-inf'))
                if input.requires_grad:
                    return MaxPlus.apply(
                                        self.in_channels,
                                        self.out_channels,
                                        input,
                                        self.kernel,
                                        self.stride,
                                        self.groups
                    )
                else:
                    return conv2d.max_plus_inference(self.in_channels,
                                                 self.out_channels,
                                                 input,
                                                 self.kernel,
                                                 self.stride,
                                                 self.groups)[0]
            case "MinPlus":
                # Pad input then pass it including kernel and padding
                input = F.pad(input, pad=self.padding, mode='constant', value=float('inf'))
                if input.requires_grad:
                    return MinPlus.apply(
                                        self.in_channels,
                                        self.out_channels,
                                        input,
                                        self.kernel,
                                        self.stride,
                                        self.groups
                    )
                else:
                    return conv2d.min_plus_inference(self.in_channels,
                                                         self.out_channels,
                                                         input,
                                                         self.kernel,
                                                         self.stride,
                                                         self.groups)[0]
            case "SmoothMax":
                input = F.pad(input, pad=self.padding, mode='constant', value=float('-inf'))
                if input.requires_grad:
                    return SmoothMax.apply(
                                        self.in_channels,
                                        self.out_channels,
                                        input,
                                        self.kernel,
                                        self.stride,
                                        self.alpha,
                                        self.groups
                    )
                else:
                    return conv2d.smooth_max_forward(self.in_channels, self.out_channels,
                                                        input.contiguous(), self.kernel.contiguous(),
                                                        self.stride, self.alpha, self.groups)[0]

            case _:
                raise ValueError(f"Expected MaxMin, MinPlus or SmoothMax but received a {self.semifield}.")

    def unsqueeze_4d(self, input):
        """
        Extends dimension of input tensor such that
        it always holds proper dimensions. This
        reduces conditional checks in code.

        Output:
        Tensor with 4D shape. Handy for C++ where a static
        size lookup is performed. Thus similar dimensions
        required. Does not affect output.
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

        Output: left, right top and bottom, where
        each variable represents the padding
        on that side of the input.
        """
        # Calculate total padding on height
        _, _, _, H = self.kernel.size()
        p_h = H - 1

        top = left = math.floor(p_h / 2)
        bottom = right = p_h - top

        return (left, right, top, bottom)

class ParametricStructuringConv(torch.nn.Module):
    """
    Function wrapper that allows a scale parameter s to be trained
    for structuring element kernel functions.

    This is an generalized way to apply these functions, but can be
    changed to suffice any needs.
    """
    def __init__(self, in_channels, out_channels, semifield_type, structuring_fn, kernel_size=3, stride=1, alpha=2.0, groups=1):
        super().__init__()
        self.s = torch.nn.Parameter(torch.tensor(1.0))
        self.structuring_fn = structuring_fn

        self.semiconv = SemiConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            semifield_type=semifield_type,
            kernel_size=kernel_size,
            stride=stride,
            alpha=alpha,
            groups=groups,
        )
    
    def forward(self, input):
        # Build kernel using current value of self.s
        kernel = self.structuring_fn(
            in_channels=self.semiconv.out_channels,
            out_channels=self.semiconv.channels_per_group,
            kernel_size=self.semiconv.kernel_size,
            s=self.s,
            device=input.device,
            dtype=input.dtype
        )

        # Set the kernel directly on the underlying SemiConv2d
        self.semiconv.kernel = kernel

        # Call the actual convolution
        return self.semiconv(input)