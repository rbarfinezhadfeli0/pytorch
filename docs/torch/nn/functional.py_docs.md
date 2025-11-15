# Documentation: `torch/nn/functional.py`

## File Metadata

- **Path**: `torch/nn/functional.py`
- **Size**: 255,743 bytes (249.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Functional interface."""

import importlib
import math
import warnings
from collections.abc import Callable
from typing import Any as _Any, Optional, TYPE_CHECKING

import torch
from torch import _VF, sym_int as _sym_int, Tensor
from torch._C import (
    _add_docstr,
    _infer_size,
    _ScalingType as ScalingType,
    _SwizzleType as SwizzleType,
)
from torch._jit_internal import (
    _overload,
    boolean_dispatch,
    BroadcastingList1,
    BroadcastingList2,
    BroadcastingList3,
)
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes
from torch.nn import _reduction as _Reduction, grad  # noqa: F401
from torch.nn.modules.utils import _list_with_default, _pair, _single, _triple
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)


# Set visibility of the bound enums to this module
ScalingType.__module__ = "torch.nn.functional"
SwizzleType.__module__ = "torch.nn.functional"

if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


conv1d = _add_docstr(
    torch.conv1d,
    r"""
conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

{tf32_note}

See :class:`~torch.nn.Conv1d` for details and output shape.

Note:
    {cudnn_reproducibility_note}

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
""".format(**reproducibility_notes, **tf32_notes)
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or
      a one-element tuple `(sW,)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a one-element tuple `(padW,)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.
    dilation: the spacing between kernel elements. Can be a single number or
      a one-element tuple `(dW,)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
      the number of groups. Default: 1

Examples::

    >>> inputs = torch.randn(33, 16, 30)
    >>> filters = torch.randn(20, 16, 5)
    >>> F.conv1d(inputs, filters)
""",
)

conv2d = _add_docstr(
    torch.conv2d,
    r"""
conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

{tf32_note}

See :class:`~torch.nn.Conv2d` for details and output shape.

Note:
    {cudnn_reproducibility_note}

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
""".format(**reproducibility_notes, **tf32_notes)
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a tuple `(padH, padW)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.

    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1
    groups: split input into groups, both :math:`\text{in\_channels}` and :math:`\text{out\_channels}`
      should be divisible by the number of groups. Default: 1

Examples::

    >>> # With square kernels and equal stride
    >>> filters = torch.randn(8, 4, 3, 3)
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> F.conv2d(inputs, filters, padding=1)
""",
)  # noqa: E501

conv3d = _add_docstr(
    torch.conv3d,
    r"""
conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 3D convolution over an input image composed of several input
planes.

{tf32_note}

See :class:`~torch.nn.Conv3d` for details and output shape.

Note:
    {cudnn_reproducibility_note}

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
""".format(**reproducibility_notes, **tf32_notes)
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a tuple `(padT, padH, padW)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.

    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dT, dH, dW)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
      the number of groups. Default: 1

Examples::

    >>> filters = torch.randn(33, 16, 3, 3, 3)
    >>> inputs = torch.randn(20, 16, 50, 10, 20)
    >>> F.conv3d(inputs, filters)
""",
)  # noqa: E501

conv_transpose1d = _add_docstr(
    torch.conv_transpose1d,
    r"""
conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

{tf32_note}

See :class:`~torch.nn.ConvTranspose1d` for details and output shape.

Note:
    {cudnn_reproducibility_note}
""".format(**reproducibility_notes, **tf32_notes)
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sW,)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padW,)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dW,)``. Default: 1

Examples::

    >>> inputs = torch.randn(20, 16, 50)
    >>> weights = torch.randn(16, 33, 5)
    >>> F.conv_transpose1d(inputs, weights)
""",
)

conv_transpose2d = _add_docstr(
    torch.conv_transpose2d,
    r"""
conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

{tf32_note}

See :class:`~torch.nn.ConvTranspose2d` for details and output shape.

Note:
    {cudnn_reproducibility_note}
""".format(**reproducibility_notes, **tf32_notes)
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sH, sW)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padH, padW)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.
      Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dH, dW)``. Default: 1

Examples::

    >>> # With square kernels and equal stride
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> weights = torch.randn(4, 8, 3, 3)
    >>> F.conv_transpose2d(inputs, weights, padding=1)
""",
)  # noqa: E501

conv_transpose3d = _add_docstr(
    torch.conv_transpose3d,
    r"""
conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 3D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution"

{tf32_note}

See :class:`~torch.nn.ConvTranspose3d` for details and output shape.

Note:
    {cudnn_reproducibility_note}
""".format(**reproducibility_notes, **tf32_notes)
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kT , kH , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sT, sH, sW)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padT, padH, padW)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple
      ``(out_padT, out_padH, out_padW)``. Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dT, dH, dW)`. Default: 1

Examples::

    >>> inputs = torch.randn(20, 16, 50, 10, 20)
    >>> weights = torch.randn(16, 33, 3, 3, 3)
    >>> F.conv_transpose3d(inputs, weights)
""",
)  # noqa: E501

conv_tbc = _add_docstr(
    torch.conv_tbc,
    r"""
Applies a 1-dimensional sequence convolution over an input sequence.
Input and output dimensions are (Time, Batch, Channels) - hence TBC.

Args:
    input: input tensor of shape :math:`(\text{sequence length} \times batch \times \text{in\_channels})`
    weight: filter of shape (:math:`\text{kernel width} \times \text{in\_channels} \times \text{out\_channels}`)
    bias: bias of shape (:math:`\text{out\_channels}`)
    pad: number of timesteps to pad. Default: 0
""",
)


# Pooling
avg_pool1d = _add_docstr(
    torch.avg_pool1d,
    r"""
avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

Applies a 1D average pooling over an input signal composed of several
input planes.

See :class:`~torch.nn.AvgPool1d` for details and output shape.

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    kernel_size: the size of the window. Can be a single number or a
      tuple `(kW,)`
    stride: the stride of the window. Can be a single number or a tuple
      `(sW,)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a single
      number or a tuple `(padW,)`. Should be at most half of effective kernel
      size, that is :math:`((kernelSize - 1) * dilation + 1) / 2`. Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` to compute the
        output shape. Default: ``False``
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation. Default: ``True``

Examples::

    >>> # pool of square window of size=3, stride=2
    >>> input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
    >>> F.avg_pool1d(input, kernel_size=3, stride=2)
    tensor([[[ 2.,  4.,  6.]]])

""",
)


avg_pool2d = _add_docstr(
    torch._C._nn.avg_pool2d,
    r"""
avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
:math:`sH \times sW` steps. The number of output features is equal to the number of
input planes.

See :class:`~torch.nn.AvgPool2d` for details and output shape.

Args:
    input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    kernel_size: size of the pooling region. Can be a single number, a single-element tuple or a
      tuple `(kH, kW)`
    stride: stride of the pooling operation. Can be a single number, a single-element tuple or a
      tuple `(sH, sW)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number, a single-element tuple or a tuple `(padH, padW)`.
      Should be at most half of effective kernel size, that
      is :math:`((kernelSize - 1) * dilation + 1) / 2`. Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` in the formula
        to compute the output shape. Default: ``False``
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation. Default: ``True``
    divisor_override: if specified, it will be used as divisor, otherwise
         size of the pooling region will be used. Default: None
""",
)

avg_pool3d = _add_docstr(
    torch._C._nn.avg_pool3d,
    r"""
avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

Applies 3D average-pooling operation in :math:`kT \times kH \times kW` regions by step
size :math:`sT \times sH \times sW` steps. The number of output features is equal to
:math:`\lfloor\frac{\text{input planes}}{sT}\rfloor`.

See :class:`~torch.nn.AvgPool3d` for details and output shape.

Args:
    input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iT \times iH , iW)`
    kernel_size: size of the pooling region. Can be a single number or a
      tuple `(kT, kH, kW)`
    stride: stride of the pooling operation. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padT, padH, padW)`. Should be at most half
      of effective kernel size, that is :math:`((kernelSize - 1) * dilation + 1) / 2`.
      Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` in the formula
        to compute the output shape
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation
    divisor_override: if specified, it will be used as divisor, otherwise
        size of the pooling region will be used. Default: None
""",
)


def fractional_max_pool2d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    output_size: Optional[BroadcastingList2[int]] = None,
    output_ratio: Optional[BroadcastingList2[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    Applies 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \times k`)
                     or a tuple `(kH, kW)`
        output_size: the target output size of the image of the form :math:`oH \times oW`.
                     Can be a tuple `(oH, oW)` or a single number :math:`oH` for a square image :math:`oH \times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool2d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> # pool of square window of size=3, and target output size 13x12
        >>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool2d_with_indices,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool2d requires specifying either an output_size or an output_ratio"
        )
    if output_size is None:
        assert output_ratio is not None
        if len(output_ratio) > 2:
            raise ValueError(
                "fractional_max_pool2d requires output_ratio to either be a single Int or tuple of Ints."
            )
        _output_ratio = _pair(output_ratio)
        output_size = [
            int(input.size(-2) * _output_ratio[0]),
            int(input.size(-1) * _output_ratio[1]),
        ]

    if _random_samples is None:
        n_batch = 1 if input.dim() == 3 else input.size(0)
        _random_samples = torch.rand(
            n_batch, input.size(-3), 2, dtype=input.dtype, device=input.device
        )
    return torch._C._nn.fractional_max_pool2d(
        input, kernel_size, output_size, _random_samples
    )


def _fractional_max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    output_size: Optional[BroadcastingList2[int]] = None,
    output_ratio: Optional[BroadcastingList2[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tensor:
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool2d,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    return fractional_max_pool2d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool2d_with_indices,
    if_false=_fractional_max_pool2d,
    module_name=__name__,
    func_name="fractional_max_pool2d",
)


def fractional_max_pool3d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = None,
    output_ratio: Optional[BroadcastingList3[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    Applies 3D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kT \times kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \times k \times k`)
                     or a tuple `(kT, kH, kW)`
        output_size: the target output size of the form :math:`oT \times oH \times oW`.
                     Can be a tuple `(oT, oH, oW)` or a single number :math:`oH` for a cubic output
                     :math:`oH \times oH \times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool3d`.

    Shape:
        - Input: :math:`(N, C, T_{in}, H_{in}, W_{in})` or :math:`(C, T_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, T_{out}, H_{out}, W_{out})` or :math:`(C, T_{out}, H_{out}, W_{out})`, where
          :math:`(T_{out}, H_{out}, W_{out})=\text{output\_size}` or
          :math:`(T_{out}, H_{out}, W_{out})=\text{output\_ratio} \times (T_{in}, H_{in}, W_{in})`

    Examples::
        >>> input = torch.randn(20, 16, 50, 32, 16)
        >>> # pool of cubic window of size=3, and target output size 13x12x11
        >>> F.fractional_max_pool3d(input, 3, output_size=(13, 12, 11))
        >>> # pool of cubic window and target output size being half of input size
        >>> F.fractional_max_pool3d(input, 3, output_ratio=(0.5, 0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool3d_with_indices,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool3d requires specifying either an output_size or an output_ratio"
        )
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _triple(output_ratio)
        output_size = [
            int(input.size(-3) * _output_ratio[0]),
            int(input.size(-2) * _output_ratio[1]),
            int(input.size(-1) * _output_ratio[2]),
        ]

    if _random_samples is None:
        n_batch = 1 if input.dim() == 4 else input.size(0)
        _random_samples = torch.rand(
            n_batch, input.size(-4), 3, dtype=input.dtype, device=input.device
        )
    return torch._C._nn.fractional_max_pool3d(
        input, kernel_size, output_size, _random_samples
    )


def _fractional_max_pool3d(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = None,
    output_ratio: Optional[BroadcastingList3[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tensor:
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool3d,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    return fractional_max_pool3d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool3d_with_indices,
    if_false=_fractional_max_pool3d,
    module_name=__name__,
    func_name="fractional_max_pool3d",
)


def max_pool1d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    dilation: BroadcastingList1[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    Applies a 1D max pooling over an input signal composed of several input
    planes.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~torch.nn.MaxPool1d`, and will change in a future release.

    See :class:`~torch.nn.MaxPool1d` for details.

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`, minibatch dim optional.
        kernel_size: the size of the window. Can be a single number or a
            tuple `(kW,)`
        stride: the stride of the window. Can be a single number or a tuple
            `(sW,)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.functional.max_unpool1d` later
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool1d_with_indices,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.max_pool1d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


def _max_pool1d(
    input: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    dilation: BroadcastingList1[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool1d,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool1d_with_indices,
    if_false=_max_pool1d,
    module_name=__name__,
    func_name="max_pool1d",
)


def max_pool2d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    Applies a 2D max pooling over an input signal composed of several input
    planes.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~torch.nn.MaxPool2d`, and will change in a future release.

    See :class:`~torch.nn.MaxPool2d` for details.

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`, minibatch dim optional.
        kernel_size: size of the pooling region. Can be a single number or a
            tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
            tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.functional.max_unpool2d` later
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool2d_with_indices,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch._C._nn.max_pool2d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


def _max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool2d,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool2d_with_indices,
    if_false=_max_pool2d,
    module_name=__name__,
    func_name="max_pool2d",
)


def max_pool3d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    dilation: BroadcastingList3[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    Applies a 3D max pooling over an input signal composed of several input
    planes.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~torch.nn.MaxPool3d`, and will change in a future release.

    See :class:`~torch.nn.MaxPool3d` for details.

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iD, iH , iW)`, minibatch dim optional.
        kernel_size: size of the pooling region. Can be a single number or a
                     tuple `(kT, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
                tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.functional.max_unpool3d` later
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool3d_with_indices,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch._C._nn.max_pool3d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


def _max_pool3d(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    dilation: BroadcastingList3[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool3d,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool3d_with_indices,
    if_false=_max_pool3d,
    module_name=__name__,
    func_name="max_pool3d",
)


def _unpool_output_size(
    input: Tensor,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    output_size: Optional[list[int]],
) -> list[int]:
    input_size = input.size()
    default_size = torch.jit.annotate(list[int], [])
    for d in range(len(kernel_size)):
        default_size.append(
            (input_size[-len(kernel_size) + d] - 1) * stride[d]
            + kernel_size[d]
            - 2 * padding[d]
        )
    if output_size is None:
        ret = default_size
    else:
        if len(output_size) == len(kernel_size) + 2:
            output_size = output_size[2:]
        if len(output_size) != len(kernel_size):
            raise ValueError(
                "output_size should be a sequence containing "
                f"{len(kernel_size)} or {len(kernel_size) + 2} elements, but it has a length of '{len(output_size)}'"
            )
        for d in range(len(kernel_size)):
            min_size = default_size[d] - stride[d]
            max_size = default_size[d] + stride[d]
            if not (min_size < output_size[d] < max_size):
                raise ValueError(
                    f'invalid output_size "{output_size}" (dim {d} must be between {min_size} and {max_size})'
                )

        ret = output_size
    return ret


def max_unpool1d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    output_size: Optional[BroadcastingList1[int]] = None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool1d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    kernel_size = _single(kernel_size)
    if stride is not None:
        _stride = _single(stride)
    else:
        _stride = kernel_size
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    if isinstance(output_size, list):
        output_size = output_size + [1]
    else:
        output_size = output_size + (1,)
    return torch._C._nn.max_unpool2d(
        input.unsqueeze(-1), indices.unsqueeze(-1), output_size
    ).squeeze(-1)


def max_unpool2d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    output_size: Optional[BroadcastingList2[int]] = None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool2d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    kernel_size = _pair(kernel_size)
    if stride is not None:
        _stride = _pair(stride)
    else:
        _stride = kernel_size
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    return torch._C._nn.max_unpool2d(input, indices, output_size)


def max_unpool3d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    output_size: Optional[BroadcastingList3[int]] = None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool3d`.

    See :class:`~torch.nn.MaxUnpool3d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool3d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    kernel_size = _triple(kernel_size)
    if stride is not None:
        _stride = _triple(stride)
    else:
        _stride = kernel_size
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    return torch._C._nn.max_unpool3d(input, indices, output_size, _stride, padding)


def lp_pool3d(
    input: Tensor,
    norm_type: int | float,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""
    Apply a 3D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    When ``ceil_mode`` is ``True``, sliding windows may go off-bounds if they start within the left
    padding or the input. Sliding windows that would start in the right padded region are ignored.

    See :class:`~torch.nn.LPPool3d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool3d,
            (input,),
            input,
            norm_type,
            kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
        )
    kd, kw, kh = _triple(kernel_size)
    if stride is not None:
        out = avg_pool3d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool3d(
            input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode
        )

    return (
        (torch.sign(out) * relu(torch.abs(out))).mul(kd * kw * kh).pow(1.0 / norm_type)
    )


def lp_pool2d(
    input: Tensor,
    norm_type: int | float,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""
    Apply a 2D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    When ``ceil_mode`` is ``True``, sliding windows may go off-bounds if they start within the left
    padding or the input. Sliding windows that would start in the right padded region are ignored.

    See :class:`~torch.nn.LPPool2d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool2d,
            (input,),
            input,
            norm_type,
            kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
        )
    kw, kh = _pair(kernel_size)
    if stride is not None:
        out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool2d(
            input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode
        )

    return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1.0 / norm_type)


def lp_pool1d(
    input: Tensor,
    norm_type: int | float,
    kernel_size: int,
    stride: Optional[BroadcastingList1[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""Apply a 1D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    When ``ceil_mode`` is ``True``, sliding windows may go off-bounds if they start within the left
    padding or the input. Sliding windows that would start in the right padded region are ignored.

    See :class:`~torch.nn.LPPool1d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool1d,
            (input,),
            input,
            norm_type,
            kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
        )
    if stride is not None:
        out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool1d(
            input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode
        )

    return (
        (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1.0 / norm_type)
    )


def adaptive_max_pool1d_with_indices(
    input: Tensor,
    output_size: BroadcastingList1[int],
    return_indices: bool = False,
) -> tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    adaptive_max_pool1d(input, output_size, return_indices=False)

    Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool1d_with_indices,
            (input,),
            input,
            output_size,
            return_indices=return_indices,
        )
    return torch.adaptive_max_pool1d(input, output_size)


def _adaptive_max_pool1d(
    input: Tensor,
    output_size: BroadcastingList1[int],
    return_indices: bool = False,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool1d,
            (input,),
            input,
            output_size,
            return_indices=return_indices,
        )
    return adaptive_max_pool1d_with_indices(input, output_size)[0]


adaptive_max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool1d_with_indices,
    if_false=_adaptive_max_pool1d,
    module_name=__name__,
    func_name="adaptive_max_pool1d",
)


def adaptive_max_pool2d_with_indices(
    input: Tensor,
    output_size: BroadcastingList2[int],
    return_indices: bool = False,
) -> tuple[Tensor, Tensor]:  # noqa: D400
    r"""adaptive_max_pool2d(input, output_size, return_indices=False)

    Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool2d_with_indices,
            (input,),
            input,
            output_size,
            return_indices=return_indices,
        )
    # pyrefly: ignore [bad-argument-type]
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool2d(input, output_size)


def _adaptive_max_pool2d(
    input: Tensor,
    output_size: BroadcastingList2[int],
    return_indices: bool = False,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool2d,
            (input,),
            input,
            output_size,
            return_indices=return_indices,
        )
    return adaptive_max_pool2d_with_indices(input, output_size)[0]


adaptive_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool2d_with_indices,
    if_false=_adaptive_max_pool2d,
    module_name=__name__,
    func_name="adaptive_max_pool2d",
)


def adaptive_max_pool3d_with_indices(
    input: Tensor,
    output_size: BroadcastingList3[int],
    return_indices: bool = False,
) -> tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    adaptive_max_pool3d(input, output_size, return_indices=False)

    Applies a 3D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool3d_with_indices,
            (input,),
            input,
            output_size,
            return_indices=return_indices,
        )
    # pyrefly: ignore [bad-argument-type]
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool3d(input, output_size)


def _adaptive_max_pool3d(
    input: Tensor,
    output_size: BroadcastingList3[int],
    return_indices: bool = False,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool3d,
            (input,),
            input,
            output_size,
            return_indices=return_indices,
        )
    return adaptive_max_pool3d_with_indices(input, output_size)[0]


adaptive_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool3d_with_indices,
    if_false=_adaptive_max_pool3d,
    module_name=__name__,
    func_name="adaptive_max_pool3d",
)


adaptive_avg_pool1d = _add_docstr(
    torch.adaptive_avg_pool1d,
    r"""
adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.

Args:
    output_size: the target output size (single integer)
""",
)


def adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    r"""Apply a 2D 
```



## High-Level Overview

"""Functional interface."""import importlibimport mathimport warningsfrom collections.abc import Callablefrom typing import Any as _Any, Optional, TYPE_CHECKINGimport torchfrom torch import _VF, sym_int as _sym_int, Tensorfrom torch._C import (    _add_docstr,    _infer_size,    _ScalingType as ScalingType,    _SwizzleType as SwizzleType,)from torch._jit_internal import (    _overload,    boolean_dispatch,    BroadcastingList1,    BroadcastingList2,    BroadcastingList3,)from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notesfrom torch.nn import _reduction as _Reduction, grad  # noqa: F401from torch.nn.modules.utils import _list_with_default, _pair, _single, _triplefrom torch.overrides import (    handle_torch_function,    has_torch_function,    has_torch_function_unary,    has_torch_function_variadic,)# Set visibility of the bound enums to this moduleScalingType.__module__ = "torch.nn.functional"SwizzleType.__module__ = "torch.nn.functional"if TYPE_CHECKING:    from torch.types import _dtype as DTypeelse:    # The JIT doesn't understand Union, nor torch.dtype here    DType = inttry:    import numpy as npexcept ModuleNotFoundError:    np = None

This Python file contains 12 class(es) and 128 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyModel`

**Functions defined**: `fractional_max_pool2d_with_indices`, `_fractional_max_pool2d`, `fractional_max_pool3d_with_indices`, `_fractional_max_pool3d`, `max_pool1d_with_indices`, `_max_pool1d`, `max_pool2d_with_indices`, `_max_pool2d`, `max_pool3d_with_indices`, `_max_pool3d`, `_unpool_output_size`, `max_unpool1d`, `max_unpool2d`, `max_unpool3d`, `lp_pool3d`, `lp_pool2d`, `lp_pool1d`, `adaptive_max_pool1d_with_indices`, `_adaptive_max_pool1d`, `adaptive_max_pool2d_with_indices`

**Key imports**: importlib, math, warnings, Callable, Any as _Any, Optional, TYPE_CHECKING, torch, _VF, sym_int as _sym_int, Tensor, reproducibility_notes, sparse_support_notes, tf32_notes, _reduction as _Reduction, grad  , _list_with_default, _pair, _single, _triple


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `importlib`
- `math`
- `warnings`
- `collections.abc`: Callable
- `typing`: Any as _Any, Optional, TYPE_CHECKING
- `torch`
- `torch._torch_docs`: reproducibility_notes, sparse_support_notes, tf32_notes
- `torch.nn`: _reduction as _Reduction, grad  
- `torch.nn.modules.utils`: _list_with_default, _pair, _single, _triple
- `torch.types`: _dtype as DType
- `numpy as np`
- `operator`: mul
- `functools`: reduce
- `cannot be top level`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/nn`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`common_types.py_docs.md`](./common_types.py_docs.md)
- [`parameter.pyi_docs.md`](./parameter.pyi_docs.md)
- [`grad.py_docs.md`](./grad.py_docs.md)
- [`_reduction.py_docs.md`](./_reduction.py_docs.md)
- [`init.py_docs.md`](./init.py_docs.md)
- [`parameter.py_docs.md`](./parameter.py_docs.md)
- [`cpp.py_docs.md`](./cpp.py_docs.md)


## Cross-References

- **File Documentation**: `functional.py_docs.md`
- **Keyword Index**: `functional.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
