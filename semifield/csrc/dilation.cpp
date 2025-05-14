#include <torch/extension.h>
#include <iostream>


at::Tensor dilation_forward(const at::Tensor& a, const at::Tensor& b){

    // Initial checks to see if sizes, types and devices are equal
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.device() == b.device());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);

    // Make elements of array a and b contiguous in memory
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();

    // Check
    std::cout << a_contig.options() << std::endl;

    // Make an empty tensor of size of a and its options
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    // Obtain pointer to first element address of a and b
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();

    // Obtain pointer to first element address of result
    float* result_ptr = result.data_ptr<float>();

    // Loop through until numel is equal to i
    for (int i = 0; i < result.numel(); i++){
      result_ptr[i] = a_ptr[i] * b_ptr[i];
      std::cout << result_ptr[i] << std::endl;
    }

  // Return the result
  return result;
  }

std::vector<at::Tensor> dilation_backward(
    const at::Tensor& grad_output,  //grad output obtained from AutoGrad
    const at::Tensor& a,
    const at::Tensor& b) {

    TORCH_CHECK(grad_output.sizes() == a.sizes());
    TORCH_CHECK(grad_output.sizes() == b.sizes());
    TORCH_CHECK(a.device() == b.device() && a.device() == grad_output.device());
    TORCH_CHECK(a.dtype() == at::kFloat && b.dtype() == at::kFloat && grad_output.dtype() == at::kFloat);

    at::Tensor grad_a = grad_output * b;
    at::Tensor grad_b = grad_output * a;

    return {grad_a, grad_b};
}


//Register the C++ functions in the torch::library
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dilation_forward, "Dilation Forward");
    m.def("backward", &dilation_backward, "Dilation Backward");
  }