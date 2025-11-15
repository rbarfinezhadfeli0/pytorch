# Documentation: `torch/ao/nn/intrinsic/qat/modules/linear_relu.py`

## File Metadata

- **Path**: `torch/ao/nn/intrinsic/qat/modules/linear_relu.py`
- **Size**: 2,184 bytes (2.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.ao.nn.intrinsic.modules.fused import _FusedModule


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfigAny


__all__ = ["LinearReLU"]


class LinearReLU(nnqat.Linear, _FusedModule):
    r"""
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.ao.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.qat.LinearReLU(20, 30)
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
        qconfig: QConfigAny = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, qconfig)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(input, self.weight_fake_quant(self.weight), self.bias))

    @classmethod
    def from_float(
        cls,
        mod: torch.nn.Module,
        use_precomputed_fake_quant: bool = False,
    ) -> LinearReLU:
        return super().from_float(mod, use_precomputed_fake_quant)  # type: ignore[no-untyped-call,no-any-return]

    def to_float(self) -> nni.LinearReLU:
        linear = torch.nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        relu = torch.nn.ReLU()
        return torch.ao.nn.intrinsic.LinearReLU(linear, relu)  # type: ignore[no-untyped-call]

```



## High-Level Overview

r"""    A LinearReLU module fused from Linear and ReLU modules, attached with    FakeQuantize modules for weight, used in    quantization aware training.    We adopt the same interface as :class:`torch.nn.Linear`.    Similar to `torch.ao.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to    default.    Attributes:        weight: fake quant module for weight    Examples::        >>> # xdoctest: +SKIP        >>> m = nn.qat.LinearReLU(20, 30)        >>> input = torch.randn(128, 20)        >>> output = m(input)        >>> print(output.size())        torch.Size([128, 30])

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LinearReLU`

**Functions defined**: `__init__`, `forward`, `from_float`, `to_float`

**Key imports**: annotations, TYPE_CHECKING, torch, torch.ao.nn.intrinsic as nni, torch.ao.nn.qat as nnqat, torch.nn.functional as F, _FusedModule, QConfigAny


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/intrinsic/qat/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TYPE_CHECKING
- `torch`
- `torch.ao.nn.intrinsic as nni`
- `torch.ao.nn.qat as nnqat`
- `torch.nn.functional as F`
- `torch.ao.nn.intrinsic.modules.fused`: _FusedModule
- `torch.ao.quantization.qconfig`: QConfigAny


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

Files in the same folder (`torch/ao/nn/intrinsic/qat/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`linear_fused.py_docs.md`](./linear_fused.py_docs.md)
- [`conv_fused.py_docs.md`](./conv_fused.py_docs.md)


## Cross-References

- **File Documentation**: `linear_relu.py_docs.md`
- **Keyword Index**: `linear_relu.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
