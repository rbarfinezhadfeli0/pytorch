# Documentation: conv.py

## File Metadata
- **Path**: `torch/ao/nn/quantized/reference/modules/conv.py`
- **Size**: 15672 bytes
- **Lines**: 518
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-defs
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t

from .utils import ReferenceQuantizedModule


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]


class _ConvNd(torch.nn.modules.conv._ConvNd, ReferenceQuantizedModule):
    """A reference version of nn.quantized.Conv2d
    we will not pack the parameters in this module, since weight packing is an
    optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
    this is useful when user want to use this module in other backends like Glow.
    """

    __annotations__ = {"bias": Optional[torch.Tensor]}
    _IS_REFERENCE = True

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return qref_conv


class Conv1d(_ConvNd, nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.Conv1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv1d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv1d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        return "QuantizedConv1d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class Conv2d(_ConvNd, nn.Conv2d):
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
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv2d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv2d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        return "QuantizedConv2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class Conv3d(_ConvNd, nn.Conv3d):
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
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.Conv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv3d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv3d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        return "QuantizedConv3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class _ConvTransposeNd(_ConvNd, torch.nn.modules.conv._ConvTransposeNd):
    """A reference version of nn.quantized.ConvTranspose2d
    we will not pack the parameters in this module, since weight packing is an
    optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
    this is useful when user want to use this module in other backends like Glow.
    """

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.output_padding,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return qref_conv


class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.ConvTranspose1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose1d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose1d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        return result

    def _get_name(self):
        return "QuantizedConvTranspose1d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)


class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.ConvTranspose2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose2d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.

        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose2d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

        return result

    def _get_name(self):
        return "QuantizedConvTranspose2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)


class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        nn.ConvTranspose3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose3d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose3d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        return result

    def _get_name(self):
        return "QuantizedConvTranspose3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 8 class(es): _ConvNd, Conv1d, Conv2d, Conv3d, _ConvTransposeNd, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d

### Functions
This file defines 26 function(s): from_float, __init__, forward, _get_name, from_float, __init__, forward, _get_name, from_float, __init__, forward, _get_name, from_float, from_float, __init__, forward, _get_name, from_float, __init__, forward, _get_name, from_float, __init__, forward, _get_name, from_float


## Key Components

The file contains 1289 words across 518 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 15672 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
