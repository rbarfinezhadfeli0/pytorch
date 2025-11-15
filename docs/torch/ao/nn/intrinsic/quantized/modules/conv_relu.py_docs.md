# Documentation: `torch/ao/nn/intrinsic/quantized/modules/conv_relu.py`

## File Metadata

- **Path**: `torch/ao/nn/intrinsic/quantized/modules/conv_relu.py`
- **Size**: 8,513 bytes (8.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
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

r"""    A ConvReLU1d module is a fused module of Conv1d and ReLU    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.    Attributes:        Same as torch.ao.nn.quantized.Conv1d

This Python file contains 3 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConvReLU1d`, `ConvReLU2d`, `ConvReLU3d`

**Functions defined**: `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`

**Key imports**: torch, torch.ao.nn.intrinsic, torch.ao.nn.intrinsic.qat, torch.ao.nn.quantized as nnq, torch.nn.functional as F, fuse_conv_bn_weights


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/intrinsic/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.nn.intrinsic`
- `torch.ao.nn.intrinsic.qat`
- `torch.ao.nn.quantized as nnq`
- `torch.nn.functional as F`
- `torch.nn.utils`: fuse_conv_bn_weights


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/ao/nn/intrinsic/quantized/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`conv_add.py_docs.md`](./conv_add.py_docs.md)
- [`bn_relu.py_docs.md`](./bn_relu.py_docs.md)
- [`linear_relu.py_docs.md`](./linear_relu.py_docs.md)


## Cross-References

- **File Documentation**: `conv_relu.py_docs.md`
- **Keyword Index**: `conv_relu.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
