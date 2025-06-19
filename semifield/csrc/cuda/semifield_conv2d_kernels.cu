#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>


template <typename scalar_t>
__global__ void max_min_cuda_forward_kernel(
    const int in_channels, const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output, scalar_t* indicees,
    int H, int W,
    int kH, int kW,
    const int pad_tl, const int pad_br,
    const int stride)
{
}

std::vector<at::Tensor> max_min_cuda_forward(
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int pad_tl, const int pad_br,
    const int stride)
    {

    at::Tensor output = torch::empty_like(input);
    at::Tensor indicees = torch::empty_like(input);

    const int threads = input.numel();
    const int blocks = ceil(threads / 128);
  
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        max_min_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            in_channels, out_channels,
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indicees.data_ptr<scalar_t>(),
            H, W,
            kH, kW,
            pad_tl, pad_br,
            stride
          );
        }
    ));

    return {output, indicees};
}

template <typename scalar_t>
__global__ void max_min_cuda_backward_kernel(
    const int in_channels, const int out_channels,
    const scalar_t* grad_output,
    const scalar_t* input,
    const scalar_t* kernel,
    const scalar_t* indicees,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int pad_tl,
    const int pad_br,
    const int stride) {
}

std::vector<at::Tensor> max_min_cuda_backward(
    const int in_channels, const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const at::Tensor& indicees,
    const int H, const int W,
    const int kH, const int kW,
    const int pad_tl, const int pad_br,
    const int stride) {

  dim3 threads(32, 32);
  dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_min_cuda_backward", ([&] {
    max_min_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
      in_channels, out_channels,
      grad_output.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      kernel.data_ptr<scalar_t>(),
      indicees.data_ptr<scalar_t>(),
      H, W,
      kH, kW,
      pad_tl, pad_br,
      stride
    );
  }));

  // Return whatever gradients you compute
  return {grad_output.clone(), input.clone()};  // example only
}
