# Documentation: `docs/torch/ao/nn/quantized/reference/modules/linear.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/nn/quantized/reference/modules/linear.py_docs.md`
- **Size**: 4,931 bytes (4.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/nn/quantized/reference/modules/linear.py`

## File Metadata

- **Path**: `torch/ao/nn/quantized/reference/modules/linear.py`
- **Size**: 2,254 bytes (2.20 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ReferenceQuantizedModule


__all__ = ["Linear"]


class Linear(nn.Linear, ReferenceQuantizedModule):
    """A reference quantized linear module that fits into the FX
    Graph Mode Quantization workflow
    activation will be floating point Tensor, we will store floating
    point weight as well in the module, but in forward we'll quantize
    and dequantize the weight before running the floating point functional
    linear operator.
    """

    _IS_REFERENCE = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias_, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self) -> str:
        return "QuantizedLinear(Reference)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.linear ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.linear --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized linear
        """
        weight_quant_dequant = self.get_weight()
        result = F.linear(x, weight_quant_dequant, self.bias)
        return result

    @classmethod
    def from_float(
        cls, float_linear: nn.Linear, weight_qparams: dict[str, Any]
    ) -> "Linear":
        qref_linear = Linear(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_linear.weight = torch.nn.Parameter(float_linear.weight.detach())
        if float_linear.bias is not None:
            qref_linear.bias = torch.nn.Parameter(float_linear.bias.detach())
        return qref_linear

```



## High-Level Overview

"""A reference quantized linear module that fits into the FX    Graph Mode Quantization workflow    activation will be floating point Tensor, we will store floating    point weight as well in the module, but in forward we'll quantize    and dequantize the weight before running the floating point functional    linear operator.

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Linear`

**Functions defined**: `__init__`, `_get_name`, `forward`, `from_float`

**Key imports**: Any, torch, torch.nn as nn, torch.nn.functional as F, ReferenceQuantizedModule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/quantized/reference/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `.utils`: ReferenceQuantizedModule


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

Files in the same folder (`torch/ao/nn/quantized/reference/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`sparse.py_docs.md`](./sparse.py_docs.md)


## Cross-References

- **File Documentation**: `linear.py_docs.md`
- **Keyword Index**: `linear.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/nn/quantized/reference/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/nn/quantized/reference/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/ao/nn/quantized/reference/modules`):

- [`sparse.py_docs.md_docs.md`](./sparse.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`sparse.py_kw.md_docs.md`](./sparse.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`rnn.py_docs.md_docs.md`](./rnn.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`conv.py_kw.md_docs.md`](./conv.py_kw.md_docs.md)
- [`linear.py_kw.md_docs.md`](./linear.py_kw.md_docs.md)
- [`conv.py_docs.md_docs.md`](./conv.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `linear.py_docs.md_docs.md`
- **Keyword Index**: `linear.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
