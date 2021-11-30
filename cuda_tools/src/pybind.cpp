#include <torch/extension.h>
#include <vector>

using namespace at;

void transform_conv_forward(Tensor input, Tensor weight, Tensor attention_score, Tensor output);

void transform_conv_backward_input(Tensor input, Tensor gradOutput, Tensor weight, Tensor attention_score, Tensor gradInput, Tensor gradAttention_score);

void transform_conv_backward_weight(Tensor input, Tensor gradOutput, Tensor attention_score, Tensor gradWeight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("transform_conv_forward", &transform_conv_forward, "transform_conv_forward (CPU/CUDA) ",
    py::arg("input"), py::arg("weight"), py::arg("attention_score"), py::arg("output"));

    m.def("transform_conv_backward_input", &transform_conv_backward_input, "transform_conv_backward_input (CPU/CUDA) ",
    py::arg("input"), py::arg("gradOutput"), py::arg("weight"), py::arg("attention_score"), py::arg("gradInput"), py::arg("gradAttention_score"));

    m.def("transform_conv_backward_weight", &transform_conv_backward_weight, "transform_conv_backward_weight (CPU/CUDA) ",
    py::arg("input"), py::arg("gradOutput"), py::arg("attention_score"), py::arg("gradWeight"));
}