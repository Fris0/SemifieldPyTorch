#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>


template <typename scalar_t>
__global__ void dilation_cuda_forward_kernel(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* result,
    int64_t size) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] * b[idx];
  }
}

at::Tensor dilation_cuda_forward(const at::Tensor& input, const at::Tensor& kernel, int top, int bottom) {
    at::Tensor result = torch::empty_like(input);
    const int threads = input.numel();
    const int blocks = ceil(threads / 32);
  
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        dilation_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            input.numel());
        }
    ));

    return result;
  }

template <typename scalar_t>
__global__ void dilation_cuda_backward_kernel(
    const scalar_t* grad_output,
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* grad_a,
    scalar_t* grad_b,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] = grad_output[idx] * b[idx];
        grad_b[idx] = grad_output[idx] * a[idx];
    }
}

  std::vector<at::Tensor> dilation_cuda_backward(const at::Tensor& grad_output, const at::Tensor& a, const at::Tensor& b, int top, int bottom) {
    auto grad_a = torch::empty_like(a);
    auto grad_b = torch::empty_like(b);
    const int threads = 1024;
    const int blocks = (a.numel() + threads - 1) / threads;
  
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "dilation_backward_cuda", ([&] {
      dilation_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.data_ptr<scalar_t>(),
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        grad_a.data_ptr<scalar_t>(),
        grad_b.data_ptr<scalar_t>(),
        a.numel());
    }));
  
    return {grad_a, grad_b};
  }