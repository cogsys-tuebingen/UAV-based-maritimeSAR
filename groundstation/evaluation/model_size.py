import numpy as np


def _conv_weights(in_channels, out_channels, kernel_size, groups=1):
    return int(out_channels * (in_channels / groups) * kernel_size[0] * kernel_size[1])


def _conv_bias(out_channels):
    return int(out_channels)


def conv_parameters(in_channels, out_channels, kernel_size, groups=1, bias=False):
    if bias:
        return int(_conv_weights(in_channels, out_channels, kernel_size, groups) + _conv_bias(out_channels))
    else:
        return int(_conv_weights(in_channels, out_channels, kernel_size, groups))


def conv_output(x, out_channels, padding, kernel_size, stride, dilation=(1, 1)):
    batch_size, channels, in_width, in_height = x
    h_out = int(((in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
    w_out = int(((in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)

    return batch_size, out_channels, h_out, w_out


def max_pool_output(x, kernel_size, padding, dilation, stride=None):
    if stride is None:
        stride = kernel_size

    batch_size, channels, in_width, in_height = x
    h_out = int((in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
    w_out = int((in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1

    return batch_size, channels, h_out, w_out


def x_to_size(x, element_size):
    from functools import reduce

    return reduce((lambda x, y: x * y), x) * element_size


def to_MB(b):
    return b / (1024 * 1024)

if __name__ == '__main__':
    element_size = 4  # single_precision

    parameters = 0
    parameters_per_layer = []
    parameters_per_layer.append(parameters)

    parameters += conv_parameters(in_channels=3, out_channels=64, kernel_size=(7, 7))
    parameters_per_layer.append(parameters)

    # layer 1
    parameters += conv_parameters(in_channels=64, out_channels=64, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=64, out_channels=64, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=64, out_channels=64, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=64, out_channels=64, kernel_size=(3, 3))
    parameters_per_layer.append(parameters)

    # layer 2
    parameters += conv_parameters(in_channels=64, out_channels=128, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=128, out_channels=128, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=128, out_channels=128, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=128, out_channels=128, kernel_size=(3, 3))
    # downsample
    parameters += conv_parameters(in_channels=64, out_channels=128, kernel_size=(1, 1))
    parameters_per_layer.append(parameters)

    # layer 3
    parameters += conv_parameters(in_channels=128, out_channels=256, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=256, out_channels=256, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=256, out_channels=256, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=256, out_channels=256, kernel_size=(3, 3))
    # downsample
    parameters += conv_parameters(in_channels=128, out_channels=256, kernel_size=(1, 1))
    parameters_per_layer.append(parameters)

    # layer 4
    parameters += conv_parameters(in_channels=256, out_channels=512, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=512, out_channels=512, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=512, out_channels=512, kernel_size=(3, 3))
    parameters += conv_parameters(in_channels=512, out_channels=512, kernel_size=(3, 3))
    # downsample
    parameters += conv_parameters(in_channels=256, out_channels=512, kernel_size=(1, 1))
    parameters_per_layer.append(parameters)

    # linear
    parameters += 512 * 1000
    parameters_per_layer.append(parameters)

    print(f"#Parameters: {parameters} => {to_MB(parameters * element_size)} MB")
    print(f"#Parameters per Layer: {parameters_per_layer} => {[to_MB(p * element_size) for p in parameters_per_layer]} MB")
    x_size = 0
    x_size_per_layer = []

    input = (1, 3, 1024, 1024)

    print(f"Input size: {to_MB(x_to_size(input, 1))} MB")
    x_size += x_to_size(input, element_size)
    x_size_per_layer.append(x_size)
    input = conv_output(input, out_channels=64, padding=(3, 3), kernel_size=(7, 7), stride=(2, 2))
    x_size += x_to_size(input, element_size)
    input = max_pool_output(input, kernel_size=(3, 3), stride=[2, 2], padding=[1, 1], dilation=[1, 1])
    x_size += x_to_size(input, element_size)
    x_size_per_layer.append(x_size)

    # layer 1
    input = conv_output(input, 64, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 64, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 64, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 64, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    x_size_per_layer.append(x_size)

    # layer 2
    input = conv_output(input, 128, padding=(1, 1), kernel_size=(3, 3), stride=(2, 2))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 128, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 128, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 128, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    x_size_per_layer.append(x_size)

    # layer 3
    input = conv_output(input, 256, padding=(1, 1), kernel_size=(3, 3), stride=(2, 2))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 256, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 256, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 256, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    x_size_per_layer.append(x_size)

    # layer 4
    input = conv_output(input, 512, padding=(1, 1), kernel_size=(3, 3), stride=(2, 2))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 512, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 512, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    input = conv_output(input, 512, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1))
    x_size += x_to_size(input, element_size)
    x_size_per_layer.append(x_size)

    input = (input[0], input[1])
    x_size += x_to_size(input, element_size)
    input = (input[0], int((input[1] / 512) * 1000))
    x_size += x_to_size(input, element_size)
    x_size_per_layer.append(x_size)

    print(f"Size of interim results: {to_MB(x_size)} MB")
    print(f"Size of interim results per layer: {[to_MB(x) for x in x_size_per_layer]} MB")

    import torchvision
    import torch

    net = torchvision.models.resnet18(pretrained=True)
    net(torch.zeros((1, 3, 1024, 1024)))
