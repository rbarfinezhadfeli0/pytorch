# Documentation: `torch/ao/nn/intrinsic/quantized/dynamic/modules/linear_relu.py`

## File Metadata

- **Path**: `torch/ao/nn/intrinsic/quantized/dynamic/modules/linear_relu.py`
- **Size**: 2,223 bytes (2.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any
from typing_extensions import Self

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized.dynamic as nnqd


__all__ = ["LinearReLU"]


class LinearReLU(nnqd.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules that can be used
    for dynamic quantization.
    Supports both, FP16 and INT8 quantization.

    We adopt the same interface as :class:`torch.ao.nn.quantized.dynamic.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.dynamic.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.quantized.dynamic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # pyrefly: ignore [bad-override]
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.qint8,
    ) -> None:
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._packed_params.dtype == torch.qint8:
            # TODO check if we should set reduce_rage = True by default here
            Y = torch.ops.quantized.linear_relu_dynamic(
                x, self._packed_params._packed_params, reduce_range=True
            )
        elif self._packed_params.dtype == torch.float16:
            Y = torch.ops.quantized.linear_relu_dynamic_fp16(
                x, self._packed_params._packed_params
            )
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear relu!")
        return Y.to(x.dtype)

    def _get_name(self) -> str:
        return "DynamicQuantizedLinearReLU"

    @classmethod
    def from_float(
        cls, mod: torch.nn.Module, use_precomputed_fake_quant: bool = False
    ) -> Self:
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qlinear_relu: Any) -> Self:  # type: ignore[override]
        return super().from_reference(ref_qlinear_relu[0])

```



## High-Level Overview

r"""    A LinearReLU module fused from Linear and ReLU modules that can be used    for dynamic quantization.    Supports both, FP16 and INT8 quantization.    We adopt the same interface as :class:`torch.ao.nn.quantized.dynamic.Linear`.    Attributes:        Same as torch.ao.nn.quantized.dynamic.Linear    Examples::        >>> # xdoctest: +SKIP        >>> m = nn.intrinsic.quantized.dynamic.LinearReLU(20, 30)        >>> input = torch.randn(128, 20)        >>> output = m(input)        >>> print(output.size())        torch.Size([128, 30])

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LinearReLU`

**Functions defined**: `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`

**Key imports**: Any, Self, torch, torch.ao.nn.intrinsic as nni, torch.ao.nn.quantized.dynamic as nnqd


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/intrinsic/quantized/dynamic/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `typing_extensions`: Self
- `torch`
- `torch.ao.nn.intrinsic as nni`
- `torch.ao.nn.quantized.dynamic as nnqd`


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

Files in the same folder (`torch/ao/nn/intrinsic/quantized/dynamic/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `linear_relu.py_docs.md`
- **Keyword Index**: `linear_relu.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
