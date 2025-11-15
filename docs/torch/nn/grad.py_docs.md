# Documentation: grad.py

## File Metadata
- **Path**: `torch/nn/grad.py`
- **Size**: 9910 bytes
- **Lines**: 298
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-defs
"""Gradient interface."""

import torch
from torch.nn.modules.utils import _pair, _single, _triple


def conv1d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv1d with respect to the input of the convolution.

    This is same as the 1D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv1d_input(input.shape, weight, grad_output)

    """
    input = grad_output.new_empty(1).expand(input_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _single(stride),
        _single(padding),
        _single(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


def conv1d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv1d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> # xdoctest: +SKIP
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv1d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _single(stride),
        _single(padding),
        _single(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]


def conv2d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv2d with respect to the input of the convolution.

    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv2d_input(input.shape, weight, grad_output)

    """
    input = grad_output.new_empty(1).expand(input_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


def conv2d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv2d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> # xdoctest: +SKIP
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]


def conv3d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv3d with respect to the input of the convolution.

    This is same as the 3D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weights tensor (out_channels x in_channels/groups x kT x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv3d_input(input.shape, weight, grad_output)

    """
    input = grad_output.new_empty(1).expand(input_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _triple(stride),
        _triple(padding),
        _triple(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


def conv3d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv3d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, weight, grad_output)
        >>> F.grad.conv3d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _triple(stride),
        _triple(padding),
        _triple(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 6 function(s): conv1d_input, conv1d_weight, conv2d_input, conv2d_weight, conv3d_input, conv3d_weight


## Key Components

The file contains 1074 words across 298 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 9910 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
