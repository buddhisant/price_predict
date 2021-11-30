import math

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from cuda_tools import ops

class transform_conv(Function):

    @staticmethod
    def forward(ctx, input, weight, attention_score):
        input=input.contiguous()
        weight=weight.contiguous()
        attention_score=attention_score.contiguous()

        ctx.save_for_backward(input, weight, attention_score)
        output = input.new_zeros((input.size(0),weight.size(0),input.size(2)))
        ops.transform_conv_forward(input, weight, attention_score, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight, attention_score = ctx.saved_tensors
        grad_output=grad_output.contiguous()

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_attention_score = torch.zeros_like(attention_score)
        ops.transform_conv_backward_input(input, grad_output, weight, attention_score, grad_input, grad_attention_score)
        ops.transform_conv_backward_weight(input, grad_output, attention_score, grad_weight)

        return grad_input, grad_weight, grad_attention_score

transform_conv1d = transform_conv.apply

class TransformConv1d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size):
        super(TransformConv1d, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size

        self.weight=nn.Parameter(torch.Tensor(out_channels,kernel_size,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.num_points
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv,stdv)

    def forward(self, input, attention_score):
        out = transform_conv1d(input, self.weight, attention_score)
        return out
