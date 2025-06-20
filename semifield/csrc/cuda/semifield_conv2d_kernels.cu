#include <torch/extension.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <vector>


template <typename scalar_t>
__global__ void max_min_cuda_forward_kernel(
    const int batch_size,
    const int in_channels, const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output, scalar_t* indicees, // These values will be changed
    int H, int W,
    int kH, int kW,
    const int pad_tl, const int pad_br,
    const int stride)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int x = idx % W;
    int y = (idx / W) % H;

    if (idx >= batch_size * W * H){
        return;
    }

    int n = idx / (H * W);
    int k_center_y = (kH % 2 == 0) ? 0 : kH / 2;
    int k_center_x = (k_center_y == 0) ? 0 : kH / 2;

    printf("%d, %d, %d\n", x, y, n);
}

std::vector<at::Tensor> max_min_cuda_forward(
    const int batch_size,
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int pad_tl, const int pad_br,
    const int stride)
    {

    at::Tensor output = torch::empty_like(input);
    at::Tensor indicees = torch::empty_like(input);

    const int threads_per_block = 64;
    const int threads = input.numel() / in_channels;
    const int blocks = (threads + threads_per_block - 1) / threads_per_block;

    printf("threads %d and blocks %d\n", threads, blocks);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        max_min_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            batch_size,
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

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

  return {grad_output, input};
}
