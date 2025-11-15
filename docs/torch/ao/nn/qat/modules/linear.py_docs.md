# Documentation: `torch/ao/nn/qat/modules/linear.py`

## File Metadata

- **Path**: `torch/ao/nn/qat/modules/linear.py`
- **Size**: 3,051 bytes (2.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.intrinsic import LinearReLU
from torch.nn.utils.parametrize import (
    is_parametrized,
    transfer_parametrizations_and_params,
    type_before_parametrizations,
)


__all__ = ["Linear"]


class Linear(nn.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module or qparams_dict
        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        assert type_before_parametrizations(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        if type_before_parametrizations(mod) == LinearReLU:
            mod = mod[0]

        qconfig = mod.qconfig
        qat_linear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            qconfig=qconfig,
        )

        if is_parametrized(mod, "weight"):
            transfer_parametrizations_and_params(mod, qat_linear, "weight")
        else:
            qat_linear.weight = mod.weight

        if is_parametrized(mod, "bias"):
            transfer_parametrizations_and_params(mod, qat_linear, "bias")
        else:
            qat_linear.bias = mod.bias

        return qat_linear

    def to_float(self):
        linear = torch.nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        linear.train(self.training)
        return linear

```



## High-Level Overview

r"""    A linear module attached with FakeQuantize modules for weight,    used for quantization aware training.    We adopt the same interface as `torch.nn.Linear`, please see    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear    for documentation.    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to    default.    Attributes:        weight: fake quant module for weight

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Linear`

**Functions defined**: `__init__`, `forward`, `from_float`, `to_float`

**Key imports**: torch, torch.nn as nn, torch.nn.functional as F, LinearReLU


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/qat/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.ao.nn.intrinsic`: LinearReLU


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
- [`conv.py_docs.md`](./conv.py_docs.md)


## Cross-References

- **File Documentation**: `linear.py_docs.md`
- **Keyword Index**: `linear.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
