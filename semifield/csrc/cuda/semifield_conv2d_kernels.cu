#include <torch/extension.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <vector>

struct OutputRegion {
    int valid_H, valid_W;
    int start_y, start_x;

    __device__ inline void decode(int idx, int batch_size, int& n, int& y, int& x) const {
        int img_size = valid_H * valid_W;
        n = idx / img_size;
        int rel = idx % img_size;
        y = (rel / valid_W) + start_y;
        x = (rel % valid_W) + start_x;
    }
};

template <typename scalar_t>
__global__ void max_min_cuda_forward_kernel(
    struct OutputRegion region,
    const int batch_size,
    const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output, scalar_t* indicees, // These values will be changed
    int kH, int kW,
    const int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * region.valid_H * region.valid_W) return;

    int n, y, x;
    region.decode(idx, batch_size, n, y, x);
    printf("x:%d and  y:%d\n", x, y);
}

std::vector<at::Tensor> max_min_cuda_forward(
    const int batch_size,
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int stride)
    {
    // Center for kernel. For odd- and even kernels.
    int k_center_y = (kH % 2 == 0) ? 0 : kH / 2;
    int k_center_x = (kW % 2 == 0) ? 0 : kW / 2;

    // Calculate the valid start locations
    int valid_y_start = k_center_y;
    int valid_y_end = H - (kH - k_center_y - 1);
    int valid_x_start = k_center_x;
    int valid_x_end = W - (kW - k_center_x - 1);

    // Calculate bounderies of H and W
    int valid_H = valid_y_end - valid_y_start;
    int valid_W = valid_x_end - valid_x_start;

    // Calculate the kernel block count and threads for each block
    int total_threads = batch_size * valid_H * valid_W;
    int threads_per_block = 64;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Create output and indicees tensor
    at::Tensor output = torch::empty({batch_size, in_channels, valid_H, valid_W}, input.options());
    at::Tensor indicees = torch::empty_like(output);

    //
    struct OutputRegion region = {valid_H, valid_W, valid_y_start, valid_x_start};

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        max_min_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            region,
            batch_size,
            out_channels,
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indicees.data_ptr<scalar_t>(),
            kH, kW,
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
