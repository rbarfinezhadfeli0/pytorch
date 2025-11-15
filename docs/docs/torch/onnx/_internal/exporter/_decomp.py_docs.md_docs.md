# Documentation: `docs/torch/onnx/_internal/exporter/_decomp.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/exporter/_decomp.py_docs.md`
- **Size**: 5,767 bytes (5.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/exporter/_decomp.py`

## File Metadata

- **Path**: `torch/onnx/_internal/exporter/_decomp.py`
- **Size**: 2,869 bytes (2.80 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
import torch._ops


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.onnx._internal.exporter import _registration


def get_onnx_implemented_overloads(
    registry: _registration.ONNXRegistry,
) -> list[_registration.TorchOp]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    registered_ops: list[_registration.TorchOp] = []
    for onnx_decomp_meta in registry.functions.values():
        assert len(onnx_decomp_meta) > 0
        # Different OnnxDecompMeta for the same TorchOp should
        # have the same fx_target.
        fx_target = onnx_decomp_meta[0].fx_target
        registered_ops.append(fx_target)
    return registered_ops


def create_onnx_friendly_decomposition_table(
    onnx_registered_ops: set[_registration.TorchOp],
) -> dict[_registration.TorchOp, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        onnx_registered_ops: All ops that have an ONNX decomposition implemented.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: dict[_registration.TorchOp, Callable] = {}

    for op_overload, decomp_fn in itertools.chain(
        torch.export.default_decompositions().items(),  # type: ignore[attr-defined]
        torch._decomp.decomposition_table.items(),  # type: ignore[attr-defined]
    ):
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        # NOTE: Do not skip torch._refs decomps. They are fine because otherwise the model is
        # not exportable anyways.
        if op_overload in onnx_registered_ops:
            continue
        # If it is HOP, we filter those out as well.
        if not hasattr(op_overload, "_schema"):
            continue
        # NOTE: torch._decomp.decomposition_table covers more ops
        # than torch.export.default_decompositions, but the latter is
        # more critical to torch.onnx.export.
        if op_overload in decomposition_table:
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table

```



## High-Level Overview

"""    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.    Args:        registry: The ONNX registry for PyTorch.    Returns:        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_onnx_implemented_overloads`, `create_onnx_friendly_decomposition_table`

**Key imports**: annotations, itertools, TYPE_CHECKING, torch, torch._ops, Callable, _registration


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `itertools`
- `typing`: TYPE_CHECKING
- `torch`
- `torch._ops`
- `collections.abc`: Callable
- `torch.onnx._internal.exporter`: _registration


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/onnx/_internal/exporter`):

- [`_registration.py_docs.md`](./_registration.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_flags.py_docs.md`](./_flags.py_docs.md)
- [`_building.py_docs.md`](./_building.py_docs.md)
- [`_ir_passes.py_docs.md`](./_ir_passes.py_docs.md)
- [`_analysis.py_docs.md`](./_analysis.py_docs.md)
- [`_verification.py_docs.md`](./_verification.py_docs.md)
- [`_capture_strategies.py_docs.md`](./_capture_strategies.py_docs.md)
- [`_tensors.py_docs.md`](./_tensors.py_docs.md)
- [`_dispatching.py_docs.md`](./_dispatching.py_docs.md)


## Cross-References

- **File Documentation**: `_decomp.py_docs.md`
- **Keyword Index**: `_decomp.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/onnx/_internal/exporter`):

- [`_onnx_program.py_docs.md_docs.md`](./_onnx_program.py_docs.md_docs.md)
- [`_testing.py_docs.md_docs.md`](./_testing.py_docs.md_docs.md)
- [`_flags.py_docs.md_docs.md`](./_flags.py_docs.md_docs.md)
- [`_verification.py_docs.md_docs.md`](./_verification.py_docs.md_docs.md)
- [`_dispatching.py_docs.md_docs.md`](./_dispatching.py_docs.md_docs.md)
- [`_errors.py_kw.md_docs.md`](./_errors.py_kw.md_docs.md)
- [`_schemas.py_kw.md_docs.md`](./_schemas.py_kw.md_docs.md)
- [`_ir_passes.py_kw.md_docs.md`](./_ir_passes.py_kw.md_docs.md)
- [`_compat.py_kw.md_docs.md`](./_compat.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_decomp.py_docs.md_docs.md`
- **Keyword Index**: `_decomp.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
