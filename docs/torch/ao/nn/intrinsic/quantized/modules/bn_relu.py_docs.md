# Documentation: `torch/ao/nn/intrinsic/quantized/modules/bn_relu.py`

## File Metadata

- **Path**: `torch/ao/nn/intrinsic/quantized/modules/bn_relu.py`
- **Size**: 3,285 bytes (3.21 KB)
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


__all__ = ["BNReLU2d", "BNReLU3d"]


class BNReLU2d(nnq.BatchNorm2d):
    r"""
    A BNReLU2d module is a fused module of BatchNorm2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm2d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU2d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        super().__init__(
            num_features, eps=eps, momentum=momentum, device=device, dtype=dtype
        )

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return torch.ops.quantized.batch_norm2d_relu(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        return "QuantizedBNReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # TODO: Add qat support for BNReLU2d
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)


class BNReLU3d(nnq.BatchNorm3d):
    r"""
    A BNReLU3d module is a fused module of BatchNorm3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm3d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm3d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        super().__init__(
            num_features, eps=eps, momentum=momentum, device=device, dtype=dtype
        )

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        return torch.ops.quantized.batch_norm3d_relu(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        return "QuantizedBNReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # TODO: Add qat support for BNReLU3d
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)

```



## High-Level Overview

r"""    A BNReLU2d module is a fused module of BatchNorm2d and ReLU    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm2d`.    Attributes:        Same as torch.ao.nn.quantized.BatchNorm2d

This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BNReLU2d`, `BNReLU3d`

**Functions defined**: `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`

**Key imports**: torch, torch.ao.nn.intrinsic, torch.ao.nn.intrinsic.qat, torch.ao.nn.quantized as nnq


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


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
- [`linear_relu.py_docs.md`](./linear_relu.py_docs.md)
- [`conv_relu.py_docs.md`](./conv_relu.py_docs.md)


## Cross-References

- **File Documentation**: `bn_relu.py_docs.md`
- **Keyword Index**: `bn_relu.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
