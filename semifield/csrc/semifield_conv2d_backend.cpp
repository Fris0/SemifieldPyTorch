#include <torch/extension.h>
#include <vector>
#include <iostream>


// Cuda dilation function declaration
std::vector<at::Tensor> max_min_cuda_forward(
    const int in_channels,
    const int out_channels,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int pad_tl,
    const int pad_br,
    const int stride);

std::vector<torch::Tensor> max_min_cuda_backward(
    const int in_channels,
    const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const at::Tensor& indicees,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int pad_tl,
    const int pad_br,
    const int stride);


// C++ implementation
std::vector<at::Tensor> max_min_forward(
                                        const int in_channels,
                                        const int out_channels,
                                        const at::Tensor& input,
                                        const at::Tensor& kernel,
                                        const int pad_tl,
                                        const int pad_br,
                                        const int stride) {
    // Get sizes of input
    auto input_sizes = input.sizes();

    std::cout << input << std::endl;
    
    const int H = input_sizes[0];
    const int W = input_sizes[1];

	// Get sizes of kernel
    auto kernel_sizes = kernel.sizes();
    const int kH = kernel_sizes[0];
    const int kW = kernel_sizes[1];



    // Return the result from the cuda kernel
    return max_min_cuda_forward(in_channels, out_channels, input, kernel, H, W, kH, kW, pad_tl, pad_br, stride);  // Returns output and indicees
}

std::vector<at::Tensor> max_min_backward(
                                        const int in_channels,
                                        const int out_channels,
                                        const at::Tensor& grad_output,
                                        const at::Tensor& input,
                                        const at::Tensor& kernel,
                                        const at::Tensor& indicees,
                                        const int pad_tl,
                                        const int pad_br,
                                        const int stride) {
    // Get sizes of input
    auto input_sizes = input.sizes();
    const int H = input_sizes[0];
    const int W = input_sizes[1];

	// Get sizes of kernel
    auto kernel_sizes = kernel.sizes();
    const int kH = kernel_sizes[0];
    const int kW = kernel_sizes[1];

    // Return the result from the cuda kernel
    return max_min_cuda_backward(in_channels, out_channels, grad_output, input, kernel, indicees, H, W, kH, kW, pad_tl, pad_br, stride);
}



//Register the C++ functions in the torch::library
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_min_forward", &max_min_forward, "Dilation Forward");
    m.def("max_min_backward", &max_min_backward, "Dilation Backward");
}