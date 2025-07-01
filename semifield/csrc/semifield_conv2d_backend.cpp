#include <torch/extension.h>
#include <vector>
#include <iostream>
#include "semifield_conv2d.h"

// Max Min
std::vector<at::Tensor> max_min_forward(const int in_channels, const int out_channels, const at::Tensor& input, const at::Tensor& kernel, const int stride) {
    // Get sizes of input
    auto input_sizes = input.sizes();

    // Batch size
    const int batch_size = input_sizes[0];

    // Input Dimensions
    const int H = input_sizes[2];
    const int W = input_sizes[3];

	// Get sizes of kernel
    auto kernel_sizes = kernel.sizes();
    const int kH = kernel_sizes[2];
    const int kW = kernel_sizes[3];

    // Return the result from the cuda kernel: output and indicees
    return max_min_cuda_forward(batch_size, in_channels, out_channels, input, kernel, H, W, kH, kW, stride);
}

std::vector<at::Tensor> max_min_backward(const int in_channels, const int out_channels, const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& input_indices, const at::Tensor& kernel_indices) {
    // Return the result from the cuda kernel
    return max_min_cuda_backward(in_channels, out_channels, grad_output, input, kernel, input_indices, kernel_indices);
}

// Min Plus
std::vector<at::Tensor> min_plus_forward(const int in_channels, const int out_channels, const at::Tensor& input, const at::Tensor& kernel, const int stride) {
    // Get sizes of input
    auto input_sizes = input.sizes();

    // Batch size
    const int batch_size = input_sizes[0];

    // Input Dimensions
    const int H = input_sizes[2];
    const int W = input_sizes[3];

	// Get sizes of kernel
    auto kernel_sizes = kernel.sizes();
    const int kH = kernel_sizes[2];
    const int kW = kernel_sizes[3];

    // Return the result from the cuda kernel: output and indicees
    return min_plus_cuda_forward(batch_size, in_channels, out_channels, input, kernel, H, W, kH, kW, stride);
}

std::vector<at::Tensor> min_plus_backward(const int in_channels, const int out_channels, const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& input_indices, const at::Tensor& kernel_indices) {
    // Return the result from the cuda kernel
    return min_plus_cuda_backward(in_channels, out_channels, grad_output, input, kernel, input_indices, kernel_indices);
}

// Smooth Max
std::vector<at::Tensor> smooth_max_forward(const int in_channels, const int out_channels, const at::Tensor& input, const at::Tensor& kernel, const int stride, const float alpha){
    // Get sizes of input
    auto input_sizes = input.sizes();

    // Batch size
    const int batch_size = input_sizes[0];

    // Input Dimensions
    const int H = input_sizes[2];
    const int W = input_sizes[3];

	// Get sizes of kernel
    auto kernel_sizes = kernel.sizes();
    const int kH = kernel_sizes[2];
    const int kW = kernel_sizes[3];

    // Return the result from the cuda kernel
    return smooth_max_cuda_forward(batch_size, in_channels, out_channels, input, kernel, H, W, kH, kW, stride, alpha);
}

std::vector<at::Tensor> smooth_max_backward(const int in_channels, const int out_channels, const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& kernel, const int stride, const float alpha){
    // Get sizes of input
    auto input_sizes = input.sizes();

    // Batch size
    const int batch_size = input_sizes[0];

    // Input Dimensions
    const int H = input_sizes[2];
    const int W = input_sizes[3];

	// Get sizes of kernel
    auto kernel_sizes = kernel.sizes();
    const int kH = kernel_sizes[2];
    const int kW = kernel_sizes[3];
    return smooth_max_cuda_backward(batch_size, in_channels, out_channels, grad_output, input, kernel, H, W, kH, kW, stride, alpha);
}

//Register the C++ functions in the torch::library
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_min_forward", &max_min_forward, "MaxMin Forward");
    m.def("max_min_backward", &max_min_backward, "MaxMin Backward");
    m.def("min_plus_forward", &min_plus_forward, "MinPlus Forward");
    m.def("min_plus_backward", &min_plus_backward, "MinPlus Backward");
    m.def("smooth_max_forward", &smooth_max_forward, "SmoothMax Forward");
    m.def("smooth_max_backward", &smooth_max_backward, "SmoothMax Backward");
}