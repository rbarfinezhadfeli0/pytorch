# Documentation: conv_relu.py

## File Metadata
- **Path**: `torch/ao/nn/intrinsic/quantized/modules/conv_relu.py`
- **Size**: 8513 bytes
- **Lines**: 267
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-defs

import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq
import torch.nn.functional as F
from torch.nn.utils import fuse_conv_bn_weights


__all__ = [
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
]

_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding


# TODO: factor out the common parts to ConvNd
class ConvReLU1d(nnq.Conv1d):
    r"""
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv1d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU1d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != "zeros":
            # Padding in Conv1d is stored as (p, p), need to get (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv1d_relu(
            input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvReLU1d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        if type(mod) is torch.ao.nn.intrinsic.qat.ConvBnReLU1d:
            assert mod.bn.running_var is not None and mod.bn.running_mean is not None
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight,
                mod.bias,
                mod.bn.running_mean,
                mod.bn.running_var,
                mod.bn.eps,
                mod.bn.weight,
                mod.bn.bias,
            )
        return super().from_float(mod, use_precomputed_fake_quant)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        assert type(ref_qconv) is not torch.ao.nn.intrinsic.ConvBnReLU1d, (
            "BatchNorm1d should be fused into Conv1d before converting to reference module"
        )
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)


class ConvReLU2d(nnq.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU2d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_relu(
            input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        if type(mod) is torch.ao.nn.intrinsic.qat.ConvBnReLU2d:
            assert mod.bn.running_var is not None and mod.bn.running_mean is not None
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight,
                mod.bias,
                mod.bn.running_mean,
                mod.bn.running_var,
                mod.bn.eps,
                mod.bn.weight,
                mod.bn.bias,
            )
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        assert type(ref_qconv) is not torch.ao.nn.intrinsic.ConvBnReLU2d, (
            "BatchNorm2d should be fused into Conv2d before converting to reference module"
        )
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)


class ConvReLU3d(nnq.Conv3d):
    r"""
    A ConvReLU3d module is a fused module of Conv3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv3d`.

    Attributes: Same as torch.ao.nn.quantized.Conv3d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU3d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        assert padding_mode != "reflect", "Conv3d does not support reflection padding"
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv3d_relu(
            input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        if type(mod) is torch.ao.nn.intrinsic.qat.ConvBnReLU3d:
            assert mod.bn.running_var is not None and mod.bn.running_mean is not None
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight,
                mod.bias,
                mod.bn.running_mean,
                mod.bn.running_var,
                mod.bn.eps,
                mod.bn.weight,
                mod.bn.bias,
            )
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        assert type(ref_qconv) is not torch.ao.nn.intrinsic.ConvBnReLU3d, (
            "BatchNorm3d should be fused into Conv3d before converting to reference module"
        )
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 3 class(es): ConvReLU1d, ConvReLU2d, ConvReLU3d

### Functions
This file defines 15 function(s): __init__, forward, _get_name, from_float, from_reference, __init__, forward, _get_name, from_float, from_reference, __init__, forward, _get_name, from_float, from_reference


## Key Components

The file contains 608 words across 267 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 8513 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
