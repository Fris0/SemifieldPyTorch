#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>


template <typename scalar_t>
__global__ void dilation_cuda_forward_kernel(const scalar_t* input, const scalar_t* kernel, scalar_t* output, scalar_t* indicees, int H, int W, int kH, int kW, int top, int bottom)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Index of flat image

    // Skip thread when flat index outside W * H
    if (idx >= W * H){
        return;
    }

    int y = idx / W;  // 2D-Coordinate for y
    int x = idx % W;  // 2D-Coordinate for x

    int relOffset = kH / 2;  // Offset around center pixel

    int maxIndex = 0;  // Max Index to store for backprop
    float maxVal = -INFINITY;  // Max value for forward pass

    for (int xOffset = -relOffset; xOffset <= relOffset; xOffset++){
        for (int yOffset = -relOffset; yOffset <= relOffset; yOffset++){
            int atX = x + xOffset;  // column
            int atY = y + yOffset;  // row

            float val = -INFINITY;

            // If row and column fall within input image
            if (atY >= 0 && atY < H && atX >= 0 && atX < W){
                int kernel_x = xOffset + relOffset;
                int kernel_y = yOffset + relOffset;

                val = input[atY * W + atX] + kernel[kernel_y * kW + kernel_x];  // Image Val + Kernel Val
            }

            if (val > maxVal){
                maxVal = val;
                maxIndex = atY * W + atX;
            }
        }
    }
    output[idx] = maxVal;
    indicees[idx] = maxIndex;
}

std::vector<at::Tensor> dilation_cuda_forward(const at::Tensor& input, const at::Tensor& kernel,
                                int H, int W,
                                int kH, int kW,
                                int top, int bottom){

    at::Tensor output = torch::empty_like(input);
    at::Tensor indicees = torch::empty_like(input);

    const int threads = input.numel();
    const int blocks = ceil(threads / 128);
  
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "dilation_forward_cuda",
    ([&] {
        dilation_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indicees.data_ptr<scalar_t>(),
            H, W,
            kH, kW,
            top, bottom
          );
        }
    ));

    return {output, indicees};
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