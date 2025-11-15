# Documentation: `docs/torch/ao/quantization/stubs.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/stubs.py_docs.md`
- **Size**: 5,028 bytes (4.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/stubs.py`

## File Metadata

- **Path**: `torch/ao/quantization/stubs.py`
- **Size**: 2,282 bytes (2.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any

import torch
from torch import nn
from torch.ao.quantization import QConfig


__all__ = ["QuantStub", "DeQuantStub", "QuantWrapper"]


class QuantStub(nn.Module):
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig: QConfig | None = None):
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DeQuantStub(nn.Module):
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig: Any | None = None):
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class QuantWrapper(nn.Module):
    r"""A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """

    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module

    def __init__(self, module: nn.Module):
        super().__init__()
        qconfig = getattr(module, "qconfig", None)
        self.add_module("quant", QuantStub(qconfig))
        self.add_module("dequant", DeQuantStub(qconfig))
        self.add_module("module", module)
        self.train(module.training)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)

```



## High-Level Overview

r"""Quantize stub module, before calibration, this is same as an observer,    it will be swapped as `nnq.Quantize` in `convert`.    Args:        qconfig: quantization configuration for the tensor,            if qconfig is not provided, we will get qconfig from parent modules

This Python file contains 4 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QuantStub`, `DeQuantStub`, `QuantWrapper`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: Any, torch, nn, QConfig


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch`
- `torch.ao.quantization`: QConfig


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/ao/quantization`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`quant_type.py_docs.md`](./quant_type.py_docs.md)
- [`fake_quantize.py_docs.md`](./fake_quantize.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`fuse_modules.py_docs.md`](./fuse_modules.py_docs.md)
- [`_equalize.py_docs.md`](./_equalize.py_docs.md)
- [`quantize.py_docs.md`](./quantize.py_docs.md)
- [`_learnable_fake_quantize.py_docs.md`](./_learnable_fake_quantize.py_docs.md)
- [`observer.py_docs.md`](./observer.py_docs.md)
- [`pattern.md_docs.md`](./pattern.md_docs.md)


## Cross-References

- **File Documentation**: `stubs.py_docs.md`
- **Keyword Index**: `stubs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/ao/quantization`):

- [`_correct_bias.py_kw.md_docs.md`](./_correct_bias.py_kw.md_docs.md)
- [`quant_type.py_kw.md_docs.md`](./quant_type.py_kw.md_docs.md)
- [`qconfig.py_docs.md_docs.md`](./qconfig.py_docs.md_docs.md)
- [`_learnable_fake_quantize.py_kw.md_docs.md`](./_learnable_fake_quantize.py_kw.md_docs.md)
- [`quantize_fx.py_kw.md_docs.md`](./quantize_fx.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`observer.py_kw.md_docs.md`](./observer.py_kw.md_docs.md)
- [`fuser_method_mappings.py_kw.md_docs.md`](./fuser_method_mappings.py_kw.md_docs.md)
- [`quantize.py_kw.md_docs.md`](./quantize.py_kw.md_docs.md)
- [`qconfig_mapping.py_kw.md_docs.md`](./qconfig_mapping.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `stubs.py_docs.md_docs.md`
- **Keyword Index**: `stubs.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
