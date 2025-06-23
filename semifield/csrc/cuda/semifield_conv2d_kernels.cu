#include <torch/extension.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <vector>

// Thread.Index output -> (x, y) in input.
struct OutputRegion {
    int output_h, output_w;
    int start_y, start_x;

    __device__ inline void decode(int idx, int batch_size, int out_channels, int stride, int& oc, int& n, int& y, int& x) const {
        int img_size = output_h * output_w;
        int rel = idx % img_size;
        oc = (idx / img_size) % out_channels;
        n = idx / img_size / out_channels;
        y = (rel / output_w) * stride + start_y;
        x = (rel % output_w) * stride + start_x;
    }
};

template <typename scalar_t>
__global__ void max_min_cuda_forward_kernel(
    struct OutputRegion region,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output, scalar_t* indicees, // These values will be changed
    int H, int W,
    int kH, int kW,
    const int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * region.output_h * region.output_w * out_channels) return;

    int oc, n, y, x;
    region.decode(idx, batch_size, out_channels, stride, oc, n, y, x);
    
    scalar_t max_val = -INFINITY;
    int max_ic = NULL;

    int k_center_y = kH / 2;
    int k_center_x = kW / 2;

    for (int ic = 0; ic < in_channels; ++ic){
        for (int dy = 0; dy < kW; ++dy){
            for (int dx = 0; dx < kH; ++dx){
                int iy = y + dy - k_center_y;
                int ix = x + dx - k_center_x;


                scalar_t val = input[n * in_channels * H * W +
                                     ic * H * W +
                                     iy * W + ix];

                scalar_t kval = kernel[oc * kH * kW +
                                       (dy + region.start_y) * kW +
                                       (dx + region.start_x)];

                scalar_t diff = val - kval;
                if (diff > max_val) {
                    max_val = diff;
                    max_ic = ic;
                }      
            }
        }
    }
    printf("oc: %d n: %d x:%d and  y:%d\n", oc, n, x, y);
    output[] = max;
}

std::vector<at::Tensor> max_min_cuda_forward(
    const int batch_size,
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int pad_w, const int pad_h,
    const int stride)
    {
    // Center for kernel. For odd- and even kernels.
    int y_start = (kH % 2 == 0) ? 0 : kH / 2;
    int x_start = (kW % 2 == 0) ? 0 : kW / 2;

    // Create output and indicees tensor
    int output_h = (H - kH) / stride + 1;
    int output_w = (W - kW) / stride + 1;

    // Calculate the kernel block count and threads for each block
    int total_threads = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Create output tensor and store indicees for backward
    at::Tensor output = torch::empty({batch_size, in_channels, out_channels, output_h, output_w}, input.options());
    at::Tensor indicees = torch::empty_like(output);

    // Use output region struct to calculate relative index
    struct OutputRegion region = {output_h, output_w, y_start, x_start};

    // [&] Take all variables that are defined above and use them in kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        max_min_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            region,
            batch_size,
            in_channels,
            out_channels,
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indicees.data_ptr<scalar_t>(),
            H, W,
            kH, kW,
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
    const scalar_t* input, const scalar_t* kernel,
    const scalar_t* indicees,
    const int H, const int W,
    const int kH, const int kW,
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
    const int pad_w, const int pad_h,
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
        stride
    );
  }));

  return {grad_output, input};
}
