# Documentation: `torch/ao/nn/qat/modules/conv.py`

## File Metadata

- **Path**: `torch/ao/nn/qat/modules/conv.py`
- **Size**: 9,801 bytes (9.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import ClassVar, Literal

import torch
import torch.nn as nn
from torch.ao.nn.intrinsic import _FusedModule
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _single, _triple


__all__ = ["Conv1d", "Conv2d", "Conv3d"]


class _ConvNd(nn.modules.conv._ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: str | tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.modules.conv._ConvNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module

        Args:
           `mod`: a float module, either produced by torch.ao.quantization utilities
           or directly from user
        """
        assert type(mod) is cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        if issubclass(type(mod), _FusedModule):
            mod = mod[0]
        qconfig = mod.qconfig
        qat_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        """This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        """
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        # conv relu
        if issubclass(cls, _FusedModule):
            modules = [conv]
            assert hasattr(cls, "_FLOAT_RELU_MODULE")
            relu = cls._FLOAT_RELU_MODULE()
            modules.append(relu)
            # pyrefly: ignore [missing-attribute]
            fused = cls._FLOAT_MODULE(*modules)
            fused.train(self.training)
            return fused
        else:
            return conv


class Conv1d(_ConvNd, nn.Conv1d):
    r"""
    A Conv1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as :class:`~torch.nn.Conv1d`

    Similar to :class:`~torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_single(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv2d(_ConvNd, nn.Conv2d):
    r"""
    A Conv2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_pair(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv3d(_ConvNd, nn.Conv3d):
    r"""
    A Conv3d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv3d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv3d#torch.nn.Conv3d
    for documentation.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_triple(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

```



## High-Level Overview


This Python file contains 4 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_ConvNd`, `Conv1d`, `Conv2d`, `Conv3d`

**Functions defined**: `__init__`, `forward`, `from_float`, `to_float`, `__init__`, `from_float`, `__init__`, `forward`, `from_float`, `__init__`, `forward`, `from_float`

**Key imports**: ClassVar, Literal, torch, torch.nn as nn, _FusedModule, _size_1_t, _size_2_t, _size_3_t, _pair, _single, _triple


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/qat/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: ClassVar, Literal
- `torch`
- `torch.nn as nn`
- `torch.ao.nn.intrinsic`: _FusedModule
- `torch.nn.common_types`: _size_1_t, _size_2_t, _size_3_t
- `torch.nn.modules.utils`: _pair, _single, _triple


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/ao/nn/qat/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`embedding_ops.py_docs.md`](./embedding_ops.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)


## Cross-References

- **File Documentation**: `conv.py_docs.md`
- **Keyword Index**: `conv.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
