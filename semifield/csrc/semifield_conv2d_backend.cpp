#include <torch/extension.h>
#include <vector>
#include <iostream>
#include "semifield_conv2d.h"

struct Conv2DParams {
    int batch_size;
    int H, W;
    int kH, kW;
};

inline Conv2DParams extract_conv2d_params(const at::Tensor& input, const at::Tensor& kernel) {
    auto input_sizes = input.sizes();
    auto kernel_sizes = kernel.sizes();

    return {
        static_cast<int>(input_sizes[0]),
        static_cast<int>(input_sizes[2]),
        static_cast<int>(input_sizes[3]),
        static_cast<int>(kernel_sizes[2]),
        static_cast<int>(kernel_sizes[3])
    };
}

// Max Min
std::vector<at::Tensor> max_min_inference(const int in_channels, const int out_channels, const at::Tensor& input, const at::Tensor& kernel, const int stride, const int groups) {
    Conv2DParams p = extract_conv2d_params(input, kernel);

    // Return the result from the cuda kernel: output and indicees
    return max_min_cuda_inference(p.batch_size, in_channels, out_channels, input, kernel, p.H, p.W, p.kH, p.kW, stride, groups);
}

std::vector<at::Tensor> max_min_forward(const int in_channels, const int out_channels, const at::Tensor& input, const at::Tensor& kernel, const int stride, const int groups) {
    Conv2DParams p = extract_conv2d_params(input, kernel);

    // Return the result from the cuda kernel: output and indicees
    return max_min_cuda_forward(p.batch_size, in_channels, out_channels, input, kernel, p.H, p.W, p.kH, p.kW, stride, groups);
}

std::vector<at::Tensor> max_min_backward(const int in_channels, const int out_channels, const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& input_indices, const at::Tensor& kernel_indices) {
    // Return the result from the cuda kernel
    return max_min_cuda_backward(in_channels, out_channels, grad_output, input, kernel, input_indices, kernel_indices);
}

// Min Plus
std::vector<at::Tensor> min_plus_inference(const int in_channels, const int out_channels, const at::Tensor& input, const at::Tensor& kernel, const int stride, const int groups) {
    Conv2DParams p = extract_conv2d_params(input, kernel);

    // Return the result from the cuda kernel: output and indicees
    return min_plus_cuda_inference(p.batch_size, in_channels, out_channels, input, kernel, p.H, p.W, p.kH, p.kW, stride, groups);
}

std::vector<at::Tensor> min_plus_forward(const int in_channels, const int out_channels, const at::Tensor& input, const at::Tensor& kernel, const int stride, const int groups) {
    Conv2DParams p = extract_conv2d_params(input, kernel);

    // Return the result from the cuda kernel: output and indicees
    return min_plus_cuda_forward(p.batch_size, in_channels, out_channels, input, kernel, p.H, p.W, p.kH, p.kW, stride, groups);
}

std::vector<at::Tensor> min_plus_backward(const int in_channels, const int out_channels, const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& input_indices, const at::Tensor& kernel_indices) {
    // Return the result from the cuda kernel
    return min_plus_cuda_backward(in_channels, out_channels, grad_output, input, kernel, input_indices, kernel_indices);
}

// Smooth Max
std::vector<at::Tensor> smooth_max_forward(const int in_channels, const int out_channels, const at::Tensor& input, const at::Tensor& kernel, const int stride, const float alpha, const int groups){
    Conv2DParams p = extract_conv2d_params(input, kernel);

    // Return the result from the cuda kernel vector with one output tensor
    return smooth_max_cuda_forward(p.batch_size, in_channels, out_channels, input, kernel, p.H, p.W, p.kH, p.kW, stride, alpha, groups);
}

std::vector<at::Tensor> smooth_max_backward(const int in_channels, const int out_channels, const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& kernel, const int stride, const float alpha, const int groups){
    Conv2DParams p = extract_conv2d_params(input, kernel);

    // Return the result from cuda
    return smooth_max_cuda_backward(p.batch_size, in_channels, out_channels, grad_output, input, kernel, p.H, p.W, p.kH, p.kW, stride, alpha, groups);
}

//Register the C++ functions in the torch::library
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_min_inference", &max_min_inference, "MaxMin Inference");
    m.def("max_min_forward", &max_min_forward, "MaxMin Forward");
    m.def("max_min_backward", &max_min_backward, "MaxMin Backward");
    m.def("min_plus_inference", &min_plus_inference, "MinPlus Inference");
    m.def("min_plus_forward", &min_plus_forward, "MinPlus Forward");
    m.def("min_plus_backward", &min_plus_backward, "MinPlus Backward");
    m.def("smooth_max_forward", &smooth_max_forward, "SmoothMax Forward");
    m.def("smooth_max_backward", &smooth_max_backward, "SmoothMax Backward");
}