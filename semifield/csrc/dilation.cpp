#include <torch/extension.h>
#include <vector>
#include <iostream>


// Cuda dilation function declaration
at::Tensor dilation_cuda_forward(
  const at::Tensor &a,
  const at::Tensor &b);

std::vector<torch::Tensor> dilation_cuda_backward(
  const at::Tensor& grad_output,
  const at::Tensor& a,
  const at::Tensor& b);


// C++ implementation

at::Tensor dilation_forward(const at::Tensor& a, const at::Tensor& b){

    // Initial checks to see if sizes, types and devices are equal
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.device() == b.device());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(a.is_cuda(), "Input tensor 'a' must be CUDA");
    TORCH_CHECK(b.is_cuda(), "Input tensor 'b' must be CUDA");

  // Return the result
  return dilation_cuda_forward(a, b);
  }

std::vector<at::Tensor> dilation_backward(
    const at::Tensor& grad_output,  //grad output obtained from AutoGrad
    const at::Tensor& a,
    const at::Tensor& b) {

    TORCH_CHECK(grad_output.sizes() == a.sizes());
    TORCH_CHECK(grad_output.sizes() == b.sizes());
    TORCH_CHECK(a.device() == b.device() && a.device() == grad_output.device());
    TORCH_CHECK(a.dtype() == at::kFloat && b.dtype() == at::kFloat && grad_output.dtype() == at::kFloat);
    TORCH_CHECK(a.is_cuda(), "Input tensor 'a' must be CUDA");
    TORCH_CHECK(b.is_cuda(), "Input tensor 'b' must be CUDA");

    return dilation_cuda_backward(grad_output, a, b);
}



//Register the C++ functions in the torch::library
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dilation_forward, "Dilation Forward");
    m.def("backward", &dilation_backward, "Dilation Backward");
  }