#include <torch/extension.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <vector>

// Thread.Index output -> (x, y) in input.
struct OutputRegion {
    int output_h, output_w;  // ouput height and width
    int start_y, start_x;    // start pos for even or odd kernel

    __device__ inline void decode(int idx, int batch_size, int out_channels, int stride, int& oc, int& n, int& y, int& x) const {
        int img_size = output_h * output_w;
        int rel = idx % img_size;
        oc = (idx / img_size) % out_channels;
        n = idx / img_size / out_channels;
        y = (rel / output_w) * stride + start_y;
        x = (rel % output_w) * stride + start_x;
    }
};

// MaxMin Inference
template <typename scalar_t>
__global__ void max_min_cuda_inference_kernel(
    struct OutputRegion region,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output,
    int H, int W,
    int kH, int kW,
    const int stride,
    const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * region.output_h * region.output_w * out_channels) return;

    // Calculate current oc, batch and x,y position in input
    int oc, n, y, x;
    region.decode(idx, batch_size, out_channels, stride, oc, n, y, x);

    // Find current group index
    int out_per_group = out_channels / groups;
    int group_idx = oc / out_per_group;

    // Calculate in_channels per group
    int in_channels_per_group = in_channels / groups;

    // Initialize values
    scalar_t max_val = std::numeric_limits<scalar_t>::lowest();

    // Find offset
    int x_offset = (kW % 2 == 0) ? 0 : kW / 2;
    int y_offset = (kH % 2 == 0) ? 0 : kH / 2;

    for (int ic = 0; ic < in_channels_per_group; ++ic){
        for (int dy = 0; dy < kH; ++dy){
            for (int dx = 0; dx < kW; ++dx){
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;

                scalar_t val = input[val_idx];
                scalar_t kval = kernel[kval_idx];

                scalar_t res = val - kval;
                if (res > max_val){
                    max_val = res;
                }
            }
        }
    }
    output[idx] = max_val;
}

std::vector<at::Tensor> max_min_cuda_inference(
    const int batch_size,
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int stride,
    const int groups)
    {
    // Center for kernel. For odd- and even kernels.
    int y_start = (kH % 2 == 0) ? 0 : kH / 2;
    int x_start = (kW % 2 == 0) ? 0 : kW / 2;

    // Create output tensor of correct height and width
    int output_h = (H - kH) / stride + 1;
    int output_w = (W - kW) / stride + 1;

    // Calculate the kernel block count and threads for each block
    int total_threads = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Create output tensor and store indicees for backward
    at::Tensor output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Use output region struct to calculate relative index
    struct OutputRegion region = {output_h, output_w, y_start, x_start};

    // [&] Take all variables that are defined above and use them in kernel
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        max_min_cuda_inference_kernel<scalar_t><<<blocks, threads_per_block>>>(
            region,
            batch_size,
            in_channels,
            out_channels,
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            H, W,
            kH, kW,
            stride,
            groups
          );
        }
    ));
    return {output};
}

// MaxMin
template <typename scalar_t>
__global__ void max_min_cuda_forward_kernel(
    struct OutputRegion region,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output,
    int* input_indices, int* kernel_indices,
    int H, int W,
    int kH, int kW,
    const int stride,
    const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * region.output_h * region.output_w * out_channels) return;

    // Calculate current oc, batch and x,y position in input
    int oc, n, y, x;
    region.decode(idx, batch_size, out_channels, stride, oc, n, y, x);

    // Find current group index
    int out_per_group = out_channels / groups;
    int group_idx = oc / out_per_group;

    // Calculate in_channels per group
    int in_channels_per_group = in_channels / groups;

    // Initialize values
    scalar_t max_val = std::numeric_limits<scalar_t>::lowest();
    int max_idx = -1;
    int max_kernel_idx = -1;

    // Find offset
    int x_offset = (kW % 2 == 0) ? 0 : kW / 2;
    int y_offset = (kH % 2 == 0) ? 0 : kH / 2;

    for (int ic = 0; ic < in_channels_per_group; ++ic){
        for (int dy = 0; dy < kH; ++dy){
            for (int dx = 0; dx < kW; ++dx){
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;

                scalar_t val = input[val_idx];
                scalar_t kval = kernel[kval_idx];

                scalar_t res = val - kval;
                if (res > max_val){
                    max_val = res;
                    max_idx = val_idx;
                    max_kernel_idx = kval_idx;
                }
            }
        }
    }
    output[idx] = max_val;
    input_indices[idx] = max_idx;
    kernel_indices[idx] = max_kernel_idx;
}

std::vector<at::Tensor> max_min_cuda_forward(
    const int batch_size,
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int stride,
    const int groups)
    {
    // Center for kernel. For odd- and even kernels.
    int y_start = (kH % 2 == 0) ? 0 : kH / 2;
    int x_start = (kW % 2 == 0) ? 0 : kW / 2;

    // Create output tensor of correct height and width
    int output_h = (H - kH) / stride + 1;
    int output_w = (W - kW) / stride + 1;

    // Calculate the kernel block count and threads for each block
    int total_threads = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Create output tensor and store indicees for backward
    at::Tensor output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    at::Tensor input_indices = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options().dtype(torch::kInt32));
    at::Tensor kernel_indices = torch::zeros_like(input_indices);

    // Use output region struct to calculate relative index
    struct OutputRegion region = {output_h, output_w, y_start, x_start};

    // [&] Take all variables that are defined above and use them in kernel
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        max_min_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            region,
            batch_size,
            in_channels,
            out_channels,
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_indices.data_ptr<int>(),
            kernel_indices.data_ptr<int>(),
            H, W,
            kH, kW,
            stride,
            groups
          );
        }
    ));
    return {output, input_indices, kernel_indices};
}

template <typename scalar_t>
__global__ void max_min_cuda_backward_kernel(
    const int in_channels, const int out_channels,
    const scalar_t* grad_output,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* grad_input, scalar_t* grad_kernel,
    const int* input_indices, const int* kernel_indices,
    const int total_threads) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads) return;

    scalar_t grad  = grad_output[idx];

    int kernel_idx = kernel_indices[idx];
    int input_idx  = input_indices[idx];

    atomicAdd(&grad_input[input_idx], grad);
    atomicAdd(&grad_kernel[kernel_idx], grad);
}

std::vector<at::Tensor> max_min_cuda_backward(
    const int in_channels, const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& kernel,
    const at::Tensor& input_indices, const at::Tensor& kernel_indices) {

    at::Tensor grad_input  = torch::zeros_like(input);
    at::Tensor grad_kernel = torch::zeros_like(kernel);

    int total_threads = grad_output.numel();
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_min_cuda_backward", ([&] {
        max_min_cuda_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            in_channels, out_channels,
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(), kernel.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(), grad_kernel.data_ptr<scalar_t>(),
            input_indices.data_ptr<int>(), kernel_indices.data_ptr<int>(),
            total_threads);
  }));

  return {grad_input, grad_kernel};
}

// Min Plus
template <typename scalar_t>
__global__ void min_plus_cuda_inference_kernel(
    struct OutputRegion region,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output,
    int H, int W,
    int kH, int kW,
    const int stride,
    const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * region.output_h * region.output_w * out_channels) return;

    // Calculate current oc, batch and x,y position in input
    int oc, n, y, x;
    region.decode(idx, batch_size, out_channels, stride, oc, n, y, x);

    // Find current group index
    int out_per_group = out_channels / groups;
    int group_idx = oc / out_per_group;

    // Calculate in_channels per group
    int in_channels_per_group = in_channels / groups;

    // Initialize values
    scalar_t min_val = std::numeric_limits<scalar_t>::max();

    // Find offset
    int x_offset = (kW % 2 == 0) ? 0 : kW / 2;
    int y_offset = (kH % 2 == 0) ? 0 : kH / 2;

    for (int ic = 0; ic < in_channels_per_group; ++ic){
        for (int dy = 0; dy < kH; ++dy){
            for (int dx = 0; dx < kW; ++dx){
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;
                scalar_t val = input[val_idx];
                scalar_t kval = kernel[kval_idx];

                scalar_t res = val + kval;
                if (res < min_val){
                    min_val = res;
                }
            }
        }
    }
    output[idx] = min_val;
}

std::vector<at::Tensor> min_plus_cuda_inference(
    const int batch_size,
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int stride,
    const int groups)
    {
    // Center for kernel. For odd- and even kernels.
    int y_start = (kH % 2 == 0) ? 0 : kH / 2;
    int x_start = (kW % 2 == 0) ? 0 : kW / 2;

    // Create output tensor of correct height and width
    int output_h = (H - kH) / stride + 1;
    int output_w = (W - kW) / stride + 1;

    // Calculate the kernel block count and threads for each block
    int total_threads = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Create output tensor and store indicees for backward
    at::Tensor output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Use output region struct to calculate relative index
    struct OutputRegion region = {output_h, output_w, y_start, x_start};

    // [&] Take all variables that are defined above and use them in kernel
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        min_plus_cuda_inference_kernel<scalar_t><<<blocks, threads_per_block>>>(
            region,
            batch_size,
            in_channels,
            out_channels,
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            H, W,
            kH, kW,
            stride,
            groups
          );
        }
    ));
    return {output};
}

template <typename scalar_t>
__global__ void min_plus_cuda_forward_kernel(
    struct OutputRegion region,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output,
    int* input_indices, int* kernel_indices,
    int H, int W,
    int kH, int kW,
    const int stride,
    const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * region.output_h * region.output_w * out_channels) return;

    // Calculate current oc, batch and x,y position in input
    int oc, n, y, x;
    region.decode(idx, batch_size, out_channels, stride, oc, n, y, x);

    // Find current group index
    int out_per_group = out_channels / groups;
    int group_idx = oc / out_per_group;

    // Calculate in_channels per group
    int in_channels_per_group = in_channels / groups;

    // Initialize values
    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    int min_idx = -1;
    int min_kernel_idx = -1;

    // Find offset
    int x_offset = (kW % 2 == 0) ? 0 : kW / 2;
    int y_offset = (kH % 2 == 0) ? 0 : kH / 2;

    for (int ic = 0; ic < in_channels_per_group; ++ic){
        for (int dy = 0; dy < kH; ++dy){
            for (int dx = 0; dx < kW; ++dx){
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;
                scalar_t val = input[val_idx];
                scalar_t kval = kernel[kval_idx];

                scalar_t res = val + kval;
                if (res < min_val){
                    min_val = res;
                    min_idx = val_idx;
                    min_kernel_idx = kval_idx;
                }
            }
        }
    }
    output[idx] = min_val;
    input_indices[idx] = min_idx;
    kernel_indices[idx] = min_kernel_idx;
}

std::vector<at::Tensor> min_plus_cuda_forward(
    const int batch_size,
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int stride,
    const int groups)
    {
    // Center for kernel. For odd- and even kernels.
    int y_start = (kH % 2 == 0) ? 0 : kH / 2;
    int x_start = (kW % 2 == 0) ? 0 : kW / 2;

    // Create output tensor of correct height and width
    int output_h = (H - kH) / stride + 1;
    int output_w = (W - kW) / stride + 1;

    // Calculate the kernel block count and threads for each block
    int total_threads = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Create output tensor and store indicees for backward
    at::Tensor output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    at::Tensor input_indices = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options().dtype(torch::kInt32));
    at::Tensor kernel_indices = torch::zeros_like(input_indices);

    // Use output region struct to calculate relative index
    struct OutputRegion region = {output_h, output_w, y_start, x_start};

    // [&] Take all variables that are defined above and use them in kernel
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        min_plus_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            region,
            batch_size,
            in_channels,
            out_channels,
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_indices.data_ptr<int>(),
            kernel_indices.data_ptr<int>(),
            H, W,
            kH, kW,
            stride,
            groups
          );
        }
    ));
    return {output, input_indices, kernel_indices};
}

template <typename scalar_t>
__global__ void min_plus_cuda_backward_kernel(
    const int in_channels, const int out_channels,
    const scalar_t* grad_output,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* grad_input, scalar_t* grad_kernel,
    const int* input_indices, const int* kernel_indices,
    const int total_threads) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads) return;

    scalar_t grad  = grad_output[idx];

    int kernel_idx = kernel_indices[idx];
    int input_idx  = input_indices[idx];

    atomicAdd(&grad_input[input_idx], grad);
    atomicAdd(&grad_kernel[kernel_idx], grad);
}

std::vector<at::Tensor> min_plus_cuda_backward(
    const int in_channels, const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& kernel,
    const at::Tensor& input_indices, const at::Tensor& kernel_indices) {

    at::Tensor grad_input  = torch::zeros_like(input);
    at::Tensor grad_kernel = torch::zeros_like(kernel);

    int total_threads = grad_output.numel();
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_min_cuda_backward", ([&] {
        min_plus_cuda_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            in_channels, out_channels,
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(), kernel.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(), grad_kernel.data_ptr<scalar_t>(),
            input_indices.data_ptr<int>(), kernel_indices.data_ptr<int>(),
            total_threads);
  }));

  return {grad_input, grad_kernel};
}

// Smooth Max
template <typename scalar_t>
__global__ void smooth_max_cuda_forward_kernel(
    struct OutputRegion region,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const scalar_t* input, const scalar_t* kernel,
    scalar_t* output,
    const scalar_t alpha,
    int H, int W,
    int kH, int kW,
    const int stride,
    const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * region.output_h * region.output_w * out_channels) return;

    // Calculate current oc, batch and x,y position in input
    int oc, n, y, x;
    region.decode(idx, batch_size, out_channels, stride, oc, n, y, x);

    // Find current group index
    int out_per_group = out_channels / groups;
    int group_idx = oc / out_per_group;

    // Calculate in_channels per group
    int in_channels_per_group = in_channels / groups;

    // Calculate offset
    int x_offset = (kW % 2 == 0) ? 0 : kW / 2;
    int y_offset = (kH % 2 == 0) ? 0 : kH / 2;

    // Find max_s for numerical stability
    scalar_t max_s = -INFINITY;
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        for (int dy = 0; dy < kH; ++dy) {
            for (int dx = 0; dx < kW; ++dx) {
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;
                scalar_t s = input[val_idx] + kernel[kval_idx];

                if (s > max_s){
                    max_s = s;
                }
            }
        }
    }

    // Accumulate exponential sum
    scalar_t sum_exp = 0.0;
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        for (int dy = 0; dy < kH; ++dy) {
            for (int dx = 0; dx < kW; ++dx) {
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;
                scalar_t s = input[val_idx] + kernel[kval_idx];
                sum_exp += expf(alpha * (s - max_s));
            }
        }
    }

    output[idx] = (1.0f / alpha) * (logf(sum_exp) + alpha * max_s);
}

// Kernel launch wrapper, adjusting from your existing function
std::vector<at::Tensor> smooth_max_cuda_forward(
    const int batch_size,
    const int in_channels, const int out_channels,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int stride,
    const float alpha,
    const int groups) {

    int y_start = (kH % 2 == 0) ? 0 : kH / 2;
    int x_start = (kW % 2 == 0) ? 0 : kW / 2;
    int output_h = (H - kH) / stride + 1;
    int output_w = (W - kW) / stride + 1;

    int total_threads = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    at::Tensor output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    struct OutputRegion region = {output_h, output_w, y_start, x_start};

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "smoothmax_cuda_forward", ([&] {
        smooth_max_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            region, batch_size, in_channels, out_channels,
            input.data_ptr<scalar_t>(), kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(alpha),
            H, W, kH, kW, stride, groups);
    }));
    return {output};
}

template <typename scalar_t>
__global__ void smooth_max_cuda_backward_kernel(
    struct OutputRegion region,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const scalar_t* grad_output,
    const scalar_t* input,
    const scalar_t* kernel,
    scalar_t* grad_input,
    scalar_t* grad_kernel,
    int H, int W,
    int kH, int kW,
    const int stride,
    const scalar_t alpha,
    const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * region.output_h * region.output_w * out_channels) return;

    // Calculate current oc, batch and x,y position in input
    int oc, n, y, x;
    region.decode(idx, batch_size, out_channels, stride, oc, n, y, x);

    // Find current group index
    int out_per_group = out_channels / groups;
    int group_idx = oc / out_per_group;

    // Calculate in_channels per group
    int in_channels_per_group = in_channels / groups;

    // Calculate offset
    int x_offset = (kW % 2 == 0) ? 0 : kW / 2;
    int y_offset = (kH % 2 == 0) ? 0 : kH / 2;

    // Find max s value for all
    scalar_t max_s = -INFINITY;
    for (int ic = 0; ic < in_channels_per_group; ++ic){
        for (int dy = 0; dy < kH; ++dy){
            for (int dx = 0; dx < kW; ++dx){
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;

                scalar_t s = kernel[kval_idx] + input[val_idx];
                if (s > max_s){
                    max_s = s;
                }
            }
        }
    }

    // Accumulate exponential sum
    scalar_t sum_exp = 0.0;
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        for (int dy = 0; dy < kH; ++dy) {
            for (int dx = 0; dx < kW; ++dx) {
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;
                scalar_t s = input[val_idx] + kernel[kval_idx];
                sum_exp += expf(alpha * (s - max_s));
            }
        }
    }

    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        for (int dy = 0; dy < kH; ++dy) {
            for (int dx = 0; dx < kW; ++dx) {
                int at_y = y + dy - y_offset;
                int at_x = x + dx - x_offset;

                int val_idx  = n * in_channels * H * W + (group_idx * in_channels_per_group + ic) * H * W + at_y * W + at_x;
                int kval_idx = oc * in_channels_per_group * kH * kW + ic * kH * kW + dy * kW + dx;

                scalar_t s = input[val_idx] + kernel[kval_idx];
                scalar_t p = expf(alpha * (s - max_s)) / sum_exp;

                scalar_t grad = grad_output[idx] * p;

                atomicAdd(&grad_input[val_idx], grad);
                atomicAdd(&grad_kernel[kval_idx], grad);
            }
        }
    }
}

std::vector<at::Tensor> smooth_max_cuda_backward(
    const int batch_size, const int in_channels, const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& kernel,
    const int H, const int W,
    const int kH, const int kW,
    const int stride,
    const float alpha,
    const int groups) {

    at::Tensor grad_input  = torch::zeros_like(input);
    at::Tensor grad_kernel = torch::zeros_like(kernel);

    int start_y = (kH % 2 == 0) ? 0 : kH / 2;
    int start_x = (kW % 2 == 0) ? 0 : kW / 2;
    int output_h = (H - kH) / stride + 1;
    int output_w = (W - kW) / stride + 1;

    int total_threads = grad_output.numel();
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    struct OutputRegion region = {output_h, output_w,
                                  start_y,
                                  start_x};

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "smoothmax_cuda_backward", ([&] {
        smooth_max_cuda_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            region,
            batch_size,
            in_channels,
            out_channels,
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            grad_kernel.data_ptr<scalar_t>(),
            H, W,
            kH, kW,
            stride, static_cast<scalar_t>(alpha),
            groups);
    }));
    return {grad_input, grad_kernel};
}
