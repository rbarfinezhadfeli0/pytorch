# Documentation: functional.py

## File Metadata
- **Path**: `torch/nn/functional.py`
- **Size**: 255743 bytes
- **Lines**: 6811
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
    r"""Apply a 2D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
    """
    if has_torch_function_unary(input):
        return handle_torch_function(adaptive_avg_pool2d, (input,), input, output_size)
    # pyrefly: ignore [bad-argument-type]
    _output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_avg_pool2d(input, _output_size)


def adaptive_avg_pool3d(input: Tensor, output_size: BroadcastingList3[int]) -> Tensor:
    r"""Apply a 3D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
    """
    if has_torch_function_unary(input):
        return handle_torch_function(adaptive_avg_pool3d, (input,), input, output_size)
    # pyrefly: ignore [bad-argument-type]
    _output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_avg_pool3d(input, _output_size)


# Activation functions
def dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

    Uses samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout, (input,), input, p=p, training=training, inplace=inplace
        )
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    return (
        _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
    )


def alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""Apply alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            alpha_dropout, (input,), input, p=p, training=training, inplace=inplace
        )
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    return (
        _VF.alpha_dropout_(input, p, training)
        if inplace
        else _VF.alpha_dropout(input, p, training)
    )


def dropout1d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly zero out entire channels (a channel is a 1D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout1d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout1d, (input,), input, p=p, training=training, inplace=inplace
        )
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    inp_dim = input.dim()
    if inp_dim not in (2, 3):
        raise RuntimeError(
            f"dropout1d: Expected 2D or 3D input, but received a {inp_dim}D input. "
            "Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
            "spatial dimension, a channel dimension, and an optional batch dimension "
            "(i.e. 2D or 3D inputs)."
        )

    is_batched = inp_dim == 3
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)

    return result


def dropout2d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly zero out entire channels (a channel is a 2D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout2d, (input,), input, p=p, training=training, inplace=inplace
        )
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    inp_dim = input.dim()
    if inp_dim not in (3, 4):
        warn_msg = (
            f"dropout2d: Received a {inp_dim}-D input to dropout2d, which is deprecated "
            "and will result in an error in a future release. To retain the behavior "
            "and silence this warning, please use dropout instead. Note that dropout2d "
            "exists to provide channel-wise dropout on inputs with 2 spatial dimensions, "
            "a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs)."
        )
        warnings.warn(warn_msg, stacklevel=2)

    # TODO: Properly support no-batch-dim inputs. For now, these are NOT supported; passing
    # a 3D input will perform dropout1d behavior instead. This was done historically and the
    # behavior is maintained here for now.
    # See https://github.com/pytorch/pytorch/issues/77081
    if inp_dim == 3:
        warnings.warn(
            "dropout2d: Received a 3D input to dropout2d and assuming that channel-wise "
            "1D dropout behavior is desired - input is interpreted as shape (N, C, L), where C "
            "is the channel dim. This behavior will change in a future release to interpret the "
            "input as one without a batch dimension, i.e. shape (C, H, W). To maintain the 1D "
            "channel-wise dropout behavior, please switch to using dropout1d instead.",
            stacklevel=2,
        )

    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    return result


def dropout3d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly zero out entire channels (a channel is a 3D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout3d, (input,), input, p=p, training=training, inplace=inplace
        )
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    inp_dim = input.dim()
    if inp_dim not in (4, 5):
        warn_msg = (
            f"dropout3d: Received a {inp_dim}-D input to dropout3d, which is deprecated "
            "and will result in an error in a future release. To retain the behavior "
            "and silence this warning, please use dropout instead. Note that dropout3d "
            "exists to provide channel-wise dropout on inputs with 3 spatial dimensions, "
            "a channel dimension, and an optional batch dimension (i.e. 4D or 5D inputs)."
        )
        warnings.warn(warn_msg, stacklevel=2)

    is_batched = inp_dim == 5
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)
    return result


def feature_alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly masks out entire channels (a channel is a feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the batch input
    is a tensor :math:`\text{input}[i, j]` of the input tensor. Instead of
    setting activations to zero, as in regular Dropout, the activations are set
    to the negative saturation value of the SELU activation function.

    Each element will be masked independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit variance.

    See :class:`~torch.nn.FeatureAlphaDropout` for details.

    Args:
        p: dropout probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            feature_alpha_dropout,
            (input,),
            input,
            p=p,
            training=training,
            inplace=inplace,
        )
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    return (
        _VF.feature_alpha_dropout_(input, p, training)
        if inplace
        else _VF.feature_alpha_dropout(input, p, training)
    )


def _threshold(
    input: Tensor,
    threshold: float,
    value: float,
    inplace: bool = False,
) -> Tensor:
    r"""Apply a threshold to each element of the input Tensor.

    See :class:`~torch.nn.Threshold` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            _threshold, (input,), input, threshold, value, inplace=inplace
        )
    if inplace:
        result = _VF.threshold_(input, threshold, value)
    else:
        result = _VF.threshold(input, threshold, value)
    return result


# We define this function as _threshold because it takes an argument
# named threshold, which clobbers the recursive reference to the
# function needed for __torch_function__ support
threshold = _threshold

threshold_ = _add_docstr(
    _VF.threshold_,
    r"""
threshold_(input, threshold, value) -> Tensor

In-place version of :func:`~threshold`.
""",
)


def relu(input: Tensor, inplace: bool = False) -> Tensor:  # noqa: D400,D402
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(relu, (input,), input, inplace=inplace)
    if inplace:
        result = torch.relu_(input)
    else:
        result = torch.relu(input)
    return result


relu_ = _add_docstr(
    torch.relu_,
    r"""
relu_(input) -> Tensor

In-place version of :func:`~relu`.
""",
)


def glu(input: Tensor, dim: int = -1) -> Tensor:  # noqa: D400,D402
    r"""
    glu(input, dim=-1) -> Tensor

    The gated linear unit. Computes:

    .. math ::
        \text{GLU}(a, b) = a \otimes \sigma(b)

    where `input` is split in half along `dim` to form `a` and `b`, :math:`\sigma`
    is the sigmoid function and :math:`\otimes` is the element-wise product between matrices.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

    Args:
        input (Tensor): input tensor
        dim (int): dimension on which to split the input. Default: -1
    """
    if has_torch_function_unary(input):
        return handle_torch_function(glu, (input,), input, dim=dim)
    if input.dim() == 0:
        raise RuntimeError(
            "glu does not support scalars because halving size must be even"
        )
    return torch._C._nn.glu(input, dim)


def hardtanh(
    input: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor:  # noqa: D400,D402
    r"""
    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
    details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            hardtanh, (input,), input, min_val=min_val, max_val=max_val, inplace=inplace
        )
    if min_val > max_val:
        raise ValueError("min_val cannot be greater than max_val")
    if inplace:
        result = torch._C._nn.hardtanh_(input, min_val, max_val)
    else:
        result = torch._C._nn.hardtanh(input, min_val, max_val)
    return result


hardtanh_ = _add_docstr(
    torch._C._nn.hardtanh_,
    r"""
hardtanh_(input, min_val=-1., max_val=1.) -> Tensor

In-place version of :func:`~hardtanh`.
""",
)


def relu6(input: Tensor, inplace: bool = False) -> Tensor:  # noqa: D400,D402
    r"""relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(relu6, (input,), input, inplace=inplace)
    if inplace:
        result = torch._C._nn.relu6_(input)
    else:
        result = torch._C._nn.relu6(input)
    return result


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""Apply the Exponential Linear Unit (ELU) function element-wise.

    See :class:`~torch.nn.ELU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(elu, (input,), input, alpha=alpha, inplace=inplace)
    if inplace:
        result = torch._C._nn.elu_(input, alpha)
    else:
        result = torch._C._nn.elu(input, alpha)
    return result


elu_ = _add_docstr(
    torch._C._nn.elu_,
    r"""
elu_(input, alpha=1.) -> Tensor

In-place version of :func:`~elu`.
""",
)


def selu(input: Tensor, inplace: bool = False) -> Tensor:  # noqa: D400,D402
    r"""selu(input, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`,
    with :math:`\alpha=1.6732632423543772848170429916717` and
    :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~torch.nn.SELU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(selu, (input,), input, inplace=inplace)
    if inplace:
        result = torch.selu_(input)
    else:
        result = torch.selu(input)
    return result


selu_ = _add_docstr(
    torch.selu_,
    r"""
selu_(input) -> Tensor

In-place version of :func:`~selu`.
""",
)


def celu(
    input: Tensor,
    alpha: float = 1.0,
    inplace: bool = False,
) -> Tensor:  # noqa: D400,D402
    r"""celu(input, alpha=1., inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))`.

    See :class:`~torch.nn.CELU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            celu, (input,), input, alpha=alpha, inplace=inplace
        )
    if inplace:
        result = torch.celu_(input, alpha)
    else:
        result = torch.celu(input, alpha)
    return result


celu_ = _add_docstr(
    torch.celu_,
    r"""
celu_(input, alpha=1.) -> Tensor

In-place version of :func:`~celu`.
""",
)


def leaky_relu(
    input: Tensor,
    negative_slope: float = 0.01,
    inplace: bool = False,
) -> Tensor:  # noqa: D400,D402
    r"""
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            leaky_relu, (input,), input, negative_slope=negative_slope, inplace=inplace
        )
    if inplace:
        result = torch._C._nn.leaky_relu_(input, negative_slope)
    else:
        result = torch._C._nn.leaky_relu(input, negative_slope)
    return result


leaky_relu_ = _add_docstr(
    torch._C._nn.leaky_relu_,
    r"""
leaky_relu_(input, negative_slope=0.01) -> Tensor

In-place version of :func:`~leaky_relu`.
""",
)


prelu = _add_docstr(
    torch.prelu,
    r"""prelu(input, weight) -> Tensor

Applies element-wise the function
:math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
learnable parameter.

.. note::
    `weight` is expected to be a scalar or 1-D tensor. If `weight` is 1-D,
    its size must match the number of input channels, determined by
    `input.size(1)` when `input.dim() >= 2`, otherwise 1.
    In the 1-D case, note that when `input` has dim > 2, `weight` can be expanded
    to the shape of `input` in a way that is not possible using normal
    :ref:`broadcasting semantics<broadcasting-semantics>`.

See :class:`~torch.nn.PReLU` for more details.
""",
)


def rrelu(
    input: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:  # noqa: D400,D402
    r"""rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor

    Randomized leaky ReLU.

    See :class:`~torch.nn.RReLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            rrelu,
            (input,),
            input,
            lower=lower,
            upper=upper,
            training=training,
            inplace=inplace,
        )
    if inplace:
        result = torch.rrelu_(input, lower, upper, training)
    else:
        result = torch.rrelu(input, lower, upper, training)
    return result


rrelu_ = _add_docstr(
    torch.rrelu_,
    r"""
rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor

In-place version of :func:`~rrelu`.
""",
)

logsigmoid = _add_docstr(
    torch._C._nn.log_sigmoid,
    r"""
logsigmoid(input) -> Tensor

Applies element-wise :math:`\text{LogSigmoid}(x_i) = \log \left(\frac{1}{1 + \exp(-x_i)}\right)`

See :class:`~torch.nn.LogSigmoid` for more details.
""",
)

gelu = _add_docstr(
    torch._C._nn.gelu,
    r"""
gelu(input, approximate = 'none') -> Tensor

When the approximate argument is 'none', it applies element-wise the function
:math:`\text{GELU}(x) = x * \Phi(x)`

where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

When the approximate argument is 'tanh', Gelu is estimated with

.. math::
    \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
""",
)

hardshrink = _add_docstr(
    torch.hardshrink,
    r"""
hardshrink(input, lambd=0.5) -> Tensor

Applies the hard shrinkage function element-wise

See :class:`~torch.nn.Hardshrink` for more details.
""",
)


def tanhshrink(input):  # noqa: D400,D402
    r"""tanhshrink(input) -> Tensor

    Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`

    See :class:`~torch.nn.Tanhshrink` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(tanhshrink, (input,), input)
    return input - input.tanh()


def softsign(input):  # noqa: D400,D402
    r"""softsign(input) -> Tensor

    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    See :class:`~torch.nn.Softsign` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(softsign, (input,), input)
    return input / (input.abs() + 1)


softplus = _add_docstr(
    torch._C._nn.softplus,
    r"""
softplus(input, beta=1, threshold=20) -> Tensor

Applies element-wise, the function :math:`\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))`.

For numerical stability the implementation reverts to the linear function
when :math:`input \times \beta > threshold`.

See :class:`~torch.nn.Softplus` for more details.
""",
)


def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        f"Implicit dimension choice for {name} has been deprecated. "
        "Change the call to include dim=X as an argument.",
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def softmin(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[DType] = None,
) -> Tensor:
    r"""Apply a softmin function.

    Note that :math:`\text{Softmin}(x) = \text{Softmax}(-x)`. See softmax definition for mathematical formula.

    See :class:`~torch.nn.Softmin` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmin will be computed (so every slice
            along dim will sum to 1).
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            softmin, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    if dim is None:
        dim = _get_softmax_dim("softmin", input.dim(), _stacklevel)
    if dtype is None:
        ret = (-input).softmax(dim)
    else:
        ret = (-input).softmax(dim, dtype=dtype)
    return ret


def softmax(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[DType] = None,
) -> Tensor:
    r"""Apply a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).

    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> Tensor:
    r"""
    Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretize.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if has_torch_function_unary(logits):
        return handle_torch_function(
            gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim
        )
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.", stacklevel=2)

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparameterization trick.
        ret = y_soft
    return ret


def log_softmax(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[DType] = None,
) -> Tensor:
    r"""Apply a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    See :class:`~torch.nn.LogSoftmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which log_softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is cast to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            log_softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    if dim is None:
        dim = _get_softmax_dim("log_softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.log_softmax(dim)
    else:
        ret = input.log_softmax(dim, dtype=dtype)
    return ret


softshrink = _add_docstr(
    torch._C._nn.softshrink,
    r"""
softshrink(input, lambd=0.5) -> Tensor

Applies the soft shrinkage function elementwise

See :class:`~torch.nn.Softshrink` for more details.
""",
)


def tanh(input):  # noqa: D400,D402
    r"""tanh(input) -> Tensor

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    return input.tanh()


def sigmoid(input):  # noqa: D400,D402
    r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    return input.sigmoid()


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Hardsigmoid function element-wise.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    See :class:`~torch.nn.Hardsigmoid` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(hardsigmoid, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.hardsigmoid_(input)
    return torch._C._nn.hardsigmoid(input)


linear = _add_docstr(
    torch._C._nn.linear,
    r"""
linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

This operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-docs>`

{sparse_beta_warning}

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Shape:

    - Input: :math:`(*, in\_features)` where `*` means any number of
      additional dimensions, including none
    - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
    - Bias: :math:`(out\_features)` or :math:`()`
    - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
""".format(**sparse_support_notes),
)


bilinear = _add_docstr(
    torch.bilinear,
    r"""
bilinear(input1, input2, weight, bias=None) -> Tensor

Applies a bilinear transformation to the incoming data:
:math:`y = x_1^T A x_2 + b`

Shape:

    - input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}`
      and :math:`*` means any number of additional dimensions.
      All but the last dimension of the inputs should be the same.
    - input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`
    - weight: :math:`(\text{out\_features}, \text{in1\_features},
      \text{in2\_features})`
    - bias: :math:`(\text{out\_features})`
    - output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
      and all but the last dimension are the same shape as the input.
""",
)


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(silu, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.silu_(input)
    return torch._C._nn.silu(input)


def mish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Mish function, element-wise.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    See :class:`~torch.nn.Mish` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(mish, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.mish_(input)
    return torch._C._nn.mish(input)


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply hardswish function, element-wise.

    Follows implementation as described in the paper:
    `Searching for MobileNetV3`_.

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    See :class:`~torch.nn.Hardswish` for more details.

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
    if has_torch_function_unary(input):
        return handle_torch_function(hardswish, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.hardswish_(input)
    return torch._C._nn.hardswish(input)


def _no_grad_embedding_renorm_(
    weight: Tensor,
    input: Tensor,
    max_norm: float,
    norm_type: float,
    # pyrefly: ignore [bad-return]
) -> tuple[Tensor, Tensor]:
    torch.embedding_renorm_(weight.detach(), input, max_norm, norm_type)


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    r"""Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`torch.nn.Embedding` for more details.

    .. note::
        Note that the analytical gradients of this function with respect to
        entries in :attr:`weight` at the row specified by :attr:`padding_idx`
        are expected to differ from the numerical ones.

    .. note::
        Note that `:class:`torch.nn.Embedding` differs from this function in
        that it initializes the row of :attr:`weight` specified by
        :attr:`padding_idx` to all zeros on construction.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad".
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
          where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> F.embedding(input, embedding_matrix)
        tensor([[[ 0.8490,  0.9625,  0.6753],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.6246,  0.9751,  0.3618],
                 [ 0.4161,  0.2419,  0.7383]],

                [[ 0.6246,  0.9751,  0.3618],
                 [ 0.0237,  0.7794,  0.0528],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.3385,  0.8612,  0.1867]]])

        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = weights
        >>> input = torch.tensor([[0, 2, 0, 5]])
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.5609,  0.5384,  0.8720],
                 [ 0.0000,  0.0000,  0.0000],
                 [ 0.6262,  0.2438,  0.7471]]])
    """
    if has_torch_function_variadic(input, weight):
        return handle_torch_function(
            embedding,
            (input, weight),
            input,
            weight,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(0), (
                "Padding_idx must be within num_embeddings"
            )
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), (
                "Padding_idx must be within num_embeddings"
            )
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        # Note [embedding_renorm contiguous]
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        # Note [embedding_renorm set_grad_enabled]
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.embedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Optional[Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tensor:
    r"""Compute sums, means or maxes of `bags` of embeddings.

    Calculation is done without instantiating the intermediate embeddings.
    See :class:`torch.nn.EmbeddingBag` for more details.

    Note:
        {backward_reproducibility_note}

    Args:
        input (LongTensor): Tensor containing bags of indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        offsets (LongTensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                             the starting index position of each bag (sequence) in :attr:`input`.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The ``p`` in the ``p``-norm to compute for the :attr:`max_norm` option.
                                     Default ``2``.
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (str, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.
                                 Note: this option is not supported when ``mode="max"``.
        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be 1. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not None.

        include_last_offset (bool, optional): if ``True``, the size of offsets is equal to the number of bags + 1.
            The last element is the size of the input, or the ending index position of the last bag (sequence).

        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the
                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated
                                     during training, i.e. it remains as a fixed "pad". Note that the embedding
                                     vector at :attr:`padding_idx` is excluded from the reduction.

    Shape:
        - :attr:`input` (LongTensor) and :attr:`offsets` (LongTensor, optional)

          - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)
            each of fixed length ``N``, and this will return ``B`` values aggregated in a way
            depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.

          - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of
            multiple bags (sequences). :attr:`offsets` is required to be a 1D tensor containing
            the starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets`
            of shape `(B)`, :attr:`input` will be viewed as having ``B`` bags.
            Empty bags (i.e., having 0-length) will have returned vectors filled by zeros.

        - :attr:`weight` (Tensor): the learnable weights of the module of shape `(num_embeddings, embedding_dim)`

        - :attr:`per_sample_weights` (Tensor, optional). Has the same shape as :attr:`input`.

        - :attr:`output`: aggregated embedding values of shape `(B, embedding_dim)`

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        >>> offsets = torch.tensor([0, 4])
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> F.embedding_bag(input, embedding_matrix, offsets)
        tensor([[ 0.3397,  0.3552,  0.5545],
                [ 0.5893,  0.4386,  0.5882]])

        >>> # example with padding_idx
        >>> embedding_matrix = torch.rand(10, 3)
        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9])
        >>> offsets = torch.tensor([0, 4])
        >>> F.embedding_bag(input, embedding_matrix, offsets, padding_idx=2, mode='sum')
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7082,  3.2145, -2.6251]])
    """
    if has_torch_function_variadic(input, weight, offsets, per_sample_weights):
        return handle_torch_function(
            embedding_bag,
            (input, weight, offsets, per_sample_weights),
            input,
            weight,
            offsets=offsets,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
    # Check for backward compatibility.
    # Used to be embedding_bag(weight, input, ...)
    # Now is     embedding_bag(input, weight, ...)
    if weight.dtype == torch.long and input.is_floating_point():
        warnings.warn(
            "Argument order of nn.functional.embedding_bag was changed. "
            "Usage `embedding_bag(weight, input, ...)` is deprecated, "
            "and should now be `embedding_bag(input, weight, ...)`.",
            stacklevel=2,
        )
        weight, input = input, weight

    if per_sample_weights is not None and input.size() != per_sample_weights.size():
        raise ValueError(
            f"embedding_bag: If per_sample_weights ({per_sample_weights.shape}) is not None, "
            f"then it must have the same shape as the input ({input.shape})"
        )

    if not weight.dim() == 2:
        raise ValueError(
            f"weight has to be a 2D Tensor, but got Tensor of dimension {weight.dim()}"
        )

    if not torch.jit.is_scripting() and input.dim() == 2 and input.is_nested:
        include_last_offset = True
        # pyrefly: ignore [missing-attribute]
        offsets = input.offsets()
        input = input.values().reshape(-1)
        if per_sample_weights is not None:
            if not per_sample_weights.is_nested:
                raise ValueError(
                    "If input is nested, then per_sample_weights must be nested if specified"
                )
            per_sample_weights = per_sample_weights.values().reshape(-1)
    elif input.dim() == 2:
        if offsets is not None:
            type_str = "<unknown>"
            # TODO: Remove this once script supports type() calls
            if not torch.jit.is_scripting():
                type_str = str(type(offsets))
            raise ValueError(
                "if input is 2D, then offsets has to be None"
                ", as input is treated is a mini-batch of"
                " fixed length sequences. However, found "
                f"offsets of type {type_str}"
            )
        offsets = torch.arange(
            0, input.numel(), input.size(1), dtype=input.dtype, device=input.device
        )

        input = input.reshape(-1)
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.reshape(-1)
    elif input.dim() == 1:
        if offsets is None:
            raise ValueError("offsets has to be a 1D Tensor but got None")
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")
    else:
        raise ValueError(
            f"input has to be 1D or 2D Tensor, but got Tensor of dimension {input.dim()}"
        )
    if mode == "sum":
        mode_enum = 0
    elif mode == "mean":
        mode_enum = 1
    elif mode == "max":
        mode_enum = 2

        if scale_grad_by_freq:
            raise ValueError(
                "max mode does not support scaling the gradient by the frequency"
            )

        if sparse:
            raise ValueError("max mode does not support sparse weights")

    else:
        raise ValueError("mode has to be one of sum, mean or max")

    if max_norm is not None:
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.nembedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)

    if per_sample_weights is not None and mode != "sum":
        raise NotImplementedError(
            "embedding_bag: per_sample_weights was not None. "
            "per_sample_weights is only supported for mode='sum' "
            f"(got mode='{mode}'). Please open a feature request on GitHub."
        )

    ret, _, _, _ = torch.embedding_bag(
        weight,
        input,
        offsets,
        scale_grad_by_freq,
        mode_enum,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )
    return ret


if embedding_bag.__doc__:
    embedding_bag.__doc__ = embedding_bag.__doc__.format(**reproducibility_notes)


def _verify_batch_size(size: list[int]) -> None:
    # XXX: JIT script does not support the reduce from functools, and mul op is a
    # builtin, which cannot be used as a value to a func yet, so rewrite this size
    # check to a simple equivalent for loop
    #
    # TODO: make use of reduce like below when JIT is ready with the mis

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 5 class(es): indices, probabilities, target, value, MyModel

### Functions
This file defines 128 function(s): fractional_max_pool2d_with_indices, _fractional_max_pool2d, fractional_max_pool3d_with_indices, _fractional_max_pool3d, max_pool1d_with_indices, _max_pool1d, max_pool2d_with_indices, _max_pool2d, max_pool3d_with_indices, _max_pool3d, _unpool_output_size, max_unpool1d, max_unpool2d, max_unpool3d, lp_pool3d, lp_pool2d, lp_pool1d, adaptive_max_pool1d_with_indices, _adaptive_max_pool1d, adaptive_max_pool2d_with_indices, _adaptive_max_pool2d, adaptive_max_pool3d_with_indices, _adaptive_max_pool3d, adaptive_avg_pool2d, adaptive_avg_pool3d, dropout, alpha_dropout, dropout1d, dropout2d, dropout3d


## Key Components

The file contains 27512 words across 6811 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 255743 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
