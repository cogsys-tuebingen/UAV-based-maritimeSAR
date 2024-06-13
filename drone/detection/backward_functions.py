import torch
import torch.nn.functional as F

import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def _grad_input_padding(grad_output, input_size, stride, padding, kernel_size, dilation=None):
    if dilation is None:
        # For backward compatibility
        dilation = [1] * len(stride)

    input_size = list(input_size)
    k = grad_output.dim() - 2

    if len(input_size) == k + 2:
        input_size = input_size[-k:]
    if len(input_size) != k:
        raise ValueError("input_size must have {} elements (got {})"
                         .format(k + 2, len(input_size)))

    def dim_size(d):
        return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] + 1
                + dilation[d] * (kernel_size[d] - 1))

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an input grad size of {}, but valid sizes range "
                 "from {} to {} (for a grad_output of {})").format(
                    input_size, min_sizes, max_sizes,
                    grad_output.size()[2:]))

    return tuple(input_size[d] - min_sizes[d] for d in range(k))


def dw_conv2d(input, grad_output, forward_layer, backward_layer):
    _pair = _ntuple(2)
    stride = forward_layer.stride
    padding = forward_layer.padding
    dilation = forward_layer.dilation
    groups = forward_layer.groups
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]

    grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1, 1)
    grad_output = grad_output.contiguous().view(
        grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
        grad_output.shape[3])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2], input.shape[3])

    backward_layer.weight.data = grad_output

    backward_layer.stride = stride
    backward_layer.padding = padding
    backward_layer.dilation = dilation
    backward_layer.groups = min_batch * in_channels

    grad_weight = backward_layer(input)

    grad_weight = grad_weight.contiguous().view(
        min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
        grad_weight.shape[3])

    kx, ky = forward_layer.kernel_size
    return grad_weight.sum(dim=0).view(
        in_channels // groups, out_channels,
        grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
        2, 0, kx).narrow(3, 0, ky)


def dx_conv2d(input, grad_output, forward_layer, backward_layer):
    input_size = input.shape
    weight = forward_layer.weight
    stride = forward_layer.stride
    padding = forward_layer.padding
    dilation = forward_layer.dilation
    groups = forward_layer.groups
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    kernel_size = (weight.shape[2], weight.shape[3])

    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                             padding, kernel_size, dilation)
    backward_layer.weight.data = weight

    backward_layer.stride = stride
    backward_layer.padding = padding
    backward_layer.dilation = dilation
    backward_layer.output_padding = grad_input_padding

    return backward_layer(grad_output)


def dx_relu(input, grad_output, forward_layer):
    input = torch.clip(torch.ceil(torch.relu(input)), min=0, max=1)
    return input * grad_output


def dx_relu(input, grad_output, forward_layer, output):
    # input = torch.clip(torch.ceil(torch.relu(input)),min=0,max=1)
    input = torch.clip(torch.ceil(output), min=0, max=1)
    return input * grad_output


def dx_batchnorm(input, grad_output, forward_layer):
    gamma = forward_layer.weight
    gamma = gamma.view(1, -1, 1, 1)
    eps = forward_layer.eps
    B = input.shape[0] * input.shape[2] * input.shape[3]

    mean = input.mean(dim=(0, 2, 3), keepdim=True)
    variance = (input ** 2).mean(dim=(0, 2, 3), keepdim=True) - mean ** 2
    # variance = input.var(dim = (0,2,3), unbiased=False, keepdim = True)
    # x_hat = (input - mean)/(torch.sqrt(variance + eps))

    dL_dxi_hat = grad_output * gamma

    dL_dvar = (-0.5 * dL_dxi_hat * (input - mean)).sum((0, 2, 3), keepdim=True) * ((variance + eps) ** -1.5)
    dL_davg = (-1.0 / torch.sqrt(variance + eps) * dL_dxi_hat).sum((0, 2, 3), keepdim=True) + (
                dL_dvar * (-2.0 * (input - mean)).sum((0, 2, 3), keepdim=True) / B)

    dL_dxi = (dL_dxi_hat / torch.sqrt(variance + eps)) + (2.0 * dL_dvar * (input - mean) / B) + (dL_davg / B)

    # dL_dgamma = (grad_output * x_hat).sum((0, 2, 3), keepdim=True)
    # dL_dbeta = (grad_output).sum((0, 2, 3), keepdim=True)

    return dL_dxi  # , dL_dgamma, dL_dbeta


def dx_dropout(input, grad_output, forward_layer, output):
    mask = torch.clip(torch.ceil(output), min=0, max=1) / forward_layer.p
    return grad_output * mask


def dx_linear(input, grad_output, forward_layer, backward_layer):
    backward_layer.weight.data = forward_layer.weight.data.T
    return backward_layer(grad_output)


def dx_avgpool(input, grad_output, forward_layer):
    # grad_channel_size = tuple(grad_output.shape[-2:]) # this is always 1x1
    input_channel_size = tuple(input.shape[-2:])
    if isinstance(input_channel_size[0], torch.Tensor):
        input_channel_size = tuple([i.item() for i in input_channel_size])

    return F.upsample(grad_output.unsqueeze(-1).unsqueeze(-1), scale_factor=input_channel_size, mode='nearest') / (
                input_channel_size[0] ** 2)
    # return F.upsample(grad_output,size=input_channel_size,mode='nearest')/(input_channel_size[0]**2)

