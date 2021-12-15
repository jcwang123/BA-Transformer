
#include <torch/extension.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor pau_cuda_forward_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

at::Tensor pau_cuda_forward_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

at::Tensor pau_cuda_forward_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

at::Tensor pau_cuda_forward_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

at::Tensor pau_cuda_forward_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

at::Tensor pau_cuda_forward_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

at::Tensor pau_cuda_forward_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);


at::Tensor pau_forward__3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_3_3(x, n, d);
}
std::vector<torch::Tensor> pau_backward__3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_3_3(grad_output, x, n, d);
}

at::Tensor pau_forward__4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_4_4(x, n, d);
}
std::vector<torch::Tensor> pau_backward__4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_4_4(grad_output, x, n, d);
}

at::Tensor pau_forward__5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_5_5(x, n, d);
}
std::vector<torch::Tensor> pau_backward__5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_5_5(grad_output, x, n, d);
}

at::Tensor pau_forward__6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_6_6(x, n, d);
}
std::vector<torch::Tensor> pau_backward__6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_6_6(grad_output, x, n, d);
}

at::Tensor pau_forward__7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_7_7(x, n, d);
}
std::vector<torch::Tensor> pau_backward__7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_7_7(grad_output, x, n, d);
}

at::Tensor pau_forward__8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_8_8(x, n, d);
}
std::vector<torch::Tensor> pau_backward__8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_8_8(grad_output, x, n, d);
}

at::Tensor pau_forward__5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_5_4(x, n, d);
}
std::vector<torch::Tensor> pau_backward__5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_5_4(grad_output, x, n, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("forward_3_3", &pau_forward__3_3, "PAU forward _3_3");
    m.def("backward_3_3", &pau_backward__3_3, "PAU backward _3_3");

    m.def("forward_4_4", &pau_forward__4_4, "PAU forward _4_4");
    m.def("backward_4_4", &pau_backward__4_4, "PAU backward _4_4");

    m.def("forward_5_5", &pau_forward__5_5, "PAU forward _5_5");
    m.def("backward_5_5", &pau_backward__5_5, "PAU backward _5_5");

    m.def("forward_6_6", &pau_forward__6_6, "PAU forward _6_6");
    m.def("backward_6_6", &pau_backward__6_6, "PAU backward _6_6");

    m.def("forward_7_7", &pau_forward__7_7, "PAU forward _7_7");
    m.def("backward_7_7", &pau_backward__7_7, "PAU backward _7_7");

    m.def("forward_8_8", &pau_forward__8_8, "PAU forward _8_8");
    m.def("backward_8_8", &pau_backward__8_8, "PAU backward _8_8");

    m.def("forward_5_4", &pau_forward__5_4, "PAU forward _5_4");
    m.def("backward_5_4", &pau_backward__5_4, "PAU backward _5_4");
}
    