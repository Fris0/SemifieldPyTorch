#include <torch/extension.h>
#include <vector>
#include <iostream>


// Cuda dilation function declaration
at::Tensor dilation_cuda_forward(
    const at::Tensor &a,
    const at::Tensor &b,
	int top,
	int bottom,
	int left,
	int right);

std::vector<torch::Tensor> dilation_cuda_backward(
    const at::Tensor& grad_output,
    const at::Tensor& a,
    const at::Tensor& b,
	int top,
	int bottom,
	int left,
	int right);


// C++ implementation

at::Tensor dilation_forward(const at::Tensor& input, const at::Tensor& kernel,
										int top, int bottom, int left, int right){
    //TORCH_CHECK();

    // Get sizes of input
    auto input_sizes = input.sizes();
    int H = input_sizes[0];
    int W = input_sizes[1];

	// Get sizes of kernel
    auto kernel_sizes = kernel.sizes();
    int kH = kernel_sizes[0];
    int kW = kernel_sizes[1];

    // Return the result from the cuda kernel
    return dilation_cuda_forward(input, kernel, top, bottom, left, right);
}

std::vector<at::Tensor> dilation_backward(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& kernel,
																					int top, int bottom, int left, int right) {
    //TORCH_CHECK()

    // Return the result from the cuda kernel
    return dilation_cuda_backward(grad_output, input, kernel, top, bottom, left, right);
}



//Register the C++ functions in the torch::library
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dilation_forward, "Dilation Forward");
    m.def("backward", &dilation_backward, "Dilation Backward");
}