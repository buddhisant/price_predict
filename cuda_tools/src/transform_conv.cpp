#include <torch/extension.h>
#include <vector>

using namespace at;

void TransformConvForwardCUDAKernalLauncher(
    Tensor input,
    Tensor weight,
    Tensor attention_score,
    Tensor output);

void TransformConvBackwardInputCUDAKernalLauncher(
    Tensor input,
    Tensor gradOutput,
    Tensor weight,
    Tensor attention_score,
    Tensor gradInput,
    Tensor gradAttention_score);

void TransformConvBackwardWeightCUDAKernalLauncher(
    Tensor input,
    Tensor gradOutput,
    Tensor attention_score,
    Tensor gradWeight);



void transform_conv_forward(Tensor input, Tensor weight, Tensor attention_score, Tensor output)
{
    TransformConvForwardCUDAKernalLauncher(input, weight, attention_score, output);
}

void transform_conv_backward_input(Tensor input, Tensor gradOutput, Tensor weight, Tensor attention_score, Tensor gradInput, Tensor gradAttention_score)
{
    TransformConvBackwardInputCUDAKernalLauncher(input, gradOutput, weight, attention_score, gradInput, gradAttention_score);
}

void transform_conv_backward_weight(Tensor input, Tensor gradOutput, Tensor attention_score, Tensor gradWeight)
{
    TransformConvBackwardWeightCUDAKernalLauncher(input, gradOutput, attention_score, gradWeight);
}
