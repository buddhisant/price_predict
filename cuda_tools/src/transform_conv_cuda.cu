#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "common_cuda_helper.hpp"

using at::Tensor;

template <typename T>
__global__ void transform_in2col_kernel(
    const int n,
    const T* data_in,
    const T* topk_score,
    const long* topk_index,
    const int kernel_size,
    const int batch_size,
    const int input_channels,
    const int length,
    T* data_col)
{
/*
n = batch_size * input_channels * length;
data_in [batch_size, input_channels, length]
topk_score [batch_size, length, kernel_size]
topk_index [batch_size, length, kernel_size]
kernel_size
batch_size
input_channels
length
data_col [kernel_size, input_channels, batch_size, length]
*/
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index<(n); index += blockDim.x * gridDim.x)
    {
        const int location_index = index % length;
        const int channel_index = (index / length)  % input_channels;
        const int batch_index = index / length / input_channels;

        T* data_col_ptr = data_col + ((channel_index * batch_size) + batch_index) * length + location_index;
        const T* data_in_ptr = data_in + (batch_index * input_channels + channel_index) * length;
        const T* topk_score_ptr = topk_score + (batch_index * length + location_index)* kernel_size;
        const long* topk_index_ptr = topk_index + (batch_index * length + location_index) * kernel_size;

        for(int i=0; i<kernel_size; i+=1)
        {
            const int cur_index = topk_index_ptr[i];
            const T cur_score = topk_score_ptr[i];
            *data_col_ptr = data_in_ptr[cur_index] * cur_score;
            data_col_ptr += batch_size * input_channels * length;
        }
    }
}

template <typename T>
__global__ void transform_col2in_kernel(
    const int n,
    const T* data_in,
    const T* data_col,
    const T* topk_score,
    const long* topk_index,
    const int kernel_size,
    const int batch_size,
    const int input_channels,
    const int length,
    T* gradAttention_score,
    T* gradInput
)
{
/*
n = batch_size * input_channels * kernel_size * length
data_in [batch_size, input_channels, length]
data_col [kernel_size, input_channels, batch_size, length]
topk_score [batch_size, length, kernel_size]
topk_index [batch_size, length, kernel_size]
kernel_size
batch_size
input_channels
length
gradAttention_score [batch_size, length, length]
gradInput [batch_size, input_channels, length]
*/
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index<(n); index += blockDim.x * gridDim.x)
    {
        const int location_index = index % length;
        const int batch_index = (index / length) % batch_size;
        const int channel_index = (index / length / batch_size) % input_channels;
        const int kernel_index = (index / length / batch_size / input_channels);

        const T cur_top_grad = data_col[index];

        const T* topk_score_ptr = topk_score + (batch_index * length + location_index)* kernel_size;
        const long* topk_index_ptr = topk_index + (batch_index * length + location_index) * kernel_size;

        const int cur_index = topk_index_ptr[kernel_index];
        const T cur_score = topk_score_ptr[kernel_index];
        const T cur_feature = data_in[(batch_index * input_channels + channel_index) * length + cur_index];

        T *location_attention = gradAttention_score + (batch_index * length + location_index) * length + cur_index;
        T *location_input = gradInput + (batch_index * input_channels + channel_index) * length + cur_index;
        atomicAdd(location_attention, cur_top_grad * cur_feature);
        atomicAdd(location_input, cur_top_grad * cur_score);
    }
}

void TransformConvForwardCUDAKernalLauncher(Tensor input, Tensor weight, Tensor attention_score, Tensor output)
{
/*
input [batch_size, input_channels, length]
weight [output_channels, kernel_size, input_channels]
attention_score [batch_size, length, length]
output [batch_size, output_channels, length]
*/
    long batch_size = input.size(0);
    long input_channels = input.size(1);
    long length = input.size(2);
    long output_channels = output.size(1);
    int kernel_size = weight.size(1);

    auto result = attention_score.topk(kernel_size);
    auto topk_score = std::get<0>(result);
    auto topk_index = std::get<1>(result);

    Tensor columns = at::zeros({kernel_size * input_channels, batch_size * length}, input.options());
    Tensor output_buffer = at::zeros({batch_size, output_channels, length},input.options());

    long num_kernels = batch_size * input_channels * length;
    const float* input_ = input.data_ptr<float>();
    const float* topk_score_ = topk_score.data_ptr<float>();
    const long* topk_index_ = topk_index.data_ptr<long>();
    float* columns_ = columns.data_ptr<float>();

    transform_in2col_kernel<<<GET_BLOCKS(num_kernels),THREADS_PER_BLOCK,0,at::cuda::getCurrentCUDAStream()>>>(
        num_kernels, input_, topk_score_, topk_index_, kernel_size, batch_size, input_channels, length, columns_
    );

    output_buffer.transpose_(0,1);
    output_buffer = output_buffer.flatten(1).addmm_(weight.flatten(1), columns).view({output_channels, batch_size, length});
    output_buffer.transpose_(0,1);

    output.copy_(output_buffer);
}

void TransformConvBackwardInputCUDAKernalLauncher(
    Tensor input,
    Tensor gradOutput,
    Tensor weight,
    Tensor attention_score,
    Tensor gradInput,
    Tensor gradAttention_score)
{
/*
input [batch_size, input_channels, length]
gradOutput [batch_size, output_channels, length]
weight [output_channels, kernel_size, input_channels]
attention_score [batch_size, length, length]
gradInput [batch_size, input_channels, length]
gradAttention_score [batch_size, length, length]
*/
    weight = weight.contiguous();
    attention_score = attention_score.contiguous();
    gradOutput = gradOutput.contiguous();
    gradInput = gradInput.contiguous();

    long batch_size = gradOutput.size(0);
    long output_channels = gradOutput.size(1);
    long length = gradOutput.size(2);
    long kernel_size = weight.size(1);
    long input_channels = weight.size(2);

    long num_kernels = batch_size * input_channels * kernel_size * length;

    Tensor columns = at::zeros({kernel_size * input_channels, batch_size * length}, gradInput.options());
    gradOutput.transpose_(0,1);
    columns = columns.addmm_(weight.flatten(1).transpose(0,1), gradOutput.flatten(1), 0.0f, 1.0f);

    auto result = attention_score.topk(kernel_size);
    auto topk_score = std::get<0>(result);
    auto topk_index = std::get<1>(result);

    const float* input_ = input.data_ptr<float>();
    const float* columns_ = columns.data_ptr<float>();
    const float* topk_score_ = topk_score.data_ptr<float>();
    const long* topk_index_ = topk_index.data_ptr<long>();
    float* gradAttention_score_ = gradAttention_score.data_ptr<float>();
    float* gradInput_ = gradInput.data_ptr<float>();

    transform_col2in_kernel<<<GET_BLOCKS(num_kernels),THREADS_PER_BLOCK,0,at::cuda::getCurrentCUDAStream()>>>(
        num_kernels, input_, columns_, topk_score_, topk_index_, kernel_size, batch_size, input_channels, length, gradAttention_score_, gradInput_
    );
    gradOutput.transpose_(0,1);
}

void TransformConvBackwardWeightCUDAKernalLauncher(Tensor input, Tensor gradOutput, Tensor attention_score, Tensor gradWeight)
{
/*
input [batch_size, input_channels, length]
gradOutput [batch_size, output_channels, length]
attention_score [batch_size, length, length]
gradWeight [output_channels, kernel_size, input_channels]
*/
    input=input.contiguous();
    gradOutput=gradOutput.contiguous();
    attention_score=attention_score.contiguous();

    long batch_size = input.size(0);
    long input_channels = input.size(1);
    long length = input.size(2);
    long output_channels = gradOutput.size(1);
    int kernel_size = gradWeight.size(1);

    auto result = attention_score.topk(kernel_size);
    auto topk_score = std::get<0>(result);
    auto topk_index = std::get<1>(result);

    Tensor columns = at::zeros({kernel_size * input_channels, batch_size * length}, input.options());
    long num_kernels = batch_size * input_channels * length;

    const float* input_ = input.data_ptr<float>();
    const float* topk_score_ = topk_score.data_ptr<float>();
    const long* topk_index_ = topk_index.data_ptr<long>();
    float* columns_ = columns.data_ptr<float>();
    transform_in2col_kernel<<<GET_BLOCKS(num_kernels),THREADS_PER_BLOCK,0,at::cuda::getCurrentCUDAStream()>>>(
        num_kernels, input_, topk_score_, topk_index_, kernel_size, batch_size, input_channels, length, columns_
    );

    Tensor gradOutputBuffer = at::zeros_like(gradOutput);
    gradOutputBuffer.copy_(gradOutput);
    gradOutputBuffer.transpose_(0, 1);
    gradWeight = gradWeight.flatten(1).addmm_(gradOutputBuffer.flatten(1),columns.transpose(0,1),0.0f,1.0f);
    gradWeight = gradWeight.view({output_channels, kernel_size, input_channels});
}