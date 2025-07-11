// Cuda dilation function declaration

std::vector<at::Tensor> max_plus_cuda_inference(
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int stride,
    const int groups);

std::vector<at::Tensor> max_plus_cuda_forward(
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int stride,
    const int groups);

std::vector<torch::Tensor> max_plus_cuda_backward(
    const int in_channels,
    const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const at::Tensor& input_indices,
    const at::Tensor& kernel_indices);

std::vector<at::Tensor> min_plus_cuda_inference(
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int stride,
    const int groups);

std::vector<at::Tensor> min_plus_cuda_forward(
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int stride,
    const int groups);

std::vector<torch::Tensor> min_plus_cuda_backward(
    const int in_channels,
    const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const at::Tensor& input_indices,
    const at::Tensor& kernel_indices);

std::vector<at::Tensor> smooth_max_cuda_forward(
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int stride,
    const float alpha,
    const int groups);

std::vector<torch::Tensor> smooth_max_cuda_backward(
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int stride,
    const float alpha,
    const int groups);