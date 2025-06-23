// Cuda dilation function declaration
std::vector<at::Tensor> max_min_cuda_forward(
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int pad_w,
    const int pad_h,
    const int stride);

std::vector<torch::Tensor> max_min_cuda_backward(
    const int in_channels,
    const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const at::Tensor& indicees,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int pad_w,
    const int pad_h,
    const int stride);
