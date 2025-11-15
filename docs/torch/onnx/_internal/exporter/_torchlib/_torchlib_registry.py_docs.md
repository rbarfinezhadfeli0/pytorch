# Documentation: `torch/onnx/_internal/exporter/_torchlib/_torchlib_registry.py`

## File Metadata

- **Path**: `torch/onnx/_internal/exporter/_torchlib/_torchlib_registry.py`
- **Size**: 2,829 bytes (2.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["onnx_impl", "get_torchlib_ops"]

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import onnxscript

import torch
from torch.onnx._internal.exporter import _constants, _registration


# Use ParamSpec for better type preservation instead of bound Callable TypeVar
_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger("__name__")


_registry: list[_registration.OnnxDecompMeta] = []


def onnx_impl(
    target: _registration.TorchOp | tuple[_registration.TorchOp, ...],
    *,
    trace_only: bool = False,
    complex: bool = False,
    opset_introduced: int = 18,
    no_compile: bool = False,
    private: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Register an ONNX implementation of a torch op."""

    if isinstance(target, torch._ops.OpOverloadPacket):
        raise TypeError(
            f"Target '{target}' should be provided as an OpOverload instead of an "
            "OpOverloadPacket. You can get the default overload with "
            "<op>.default"
        )

    def wrapper(
        func: Callable[_P, _R],
    ) -> Callable[_P, _R]:
        processed_func: Any
        if no_compile:
            processed_func = func
        else:
            torchlib_opset = onnxscript.values.Opset(
                domain=_constants.TORCHLIB_DOMAIN, version=1
            )

            if not trace_only:
                # Compile the function
                processed_func = onnxscript.script(opset=torchlib_opset)(func)
            else:
                processed_func = onnxscript.TracedOnnxFunction(torchlib_opset, func)

        if not private:
            # TODO(justinchuby): Simplify the logic and remove the private attribute
            # Skip registration if private
            if not isinstance(target, Sequence):
                targets = (target,)
            else:
                targets = target  # type: ignore[assignment]

            for t in targets:
                _registry.append(
                    _registration.OnnxDecompMeta(
                        onnx_function=processed_func,
                        fx_target=t,
                        signature=None,
                        is_complex=complex,
                        opset_introduced=opset_introduced,
                        skip_signature_inference=no_compile,
                    )
                )
        return processed_func  # type: ignore[return-value]

    return wrapper


def get_torchlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]:
    # Trigger op registration
    from torch.onnx._internal.exporter._torchlib import ops

    del ops
    assert len(_registry) != 0
    return tuple(_registry)

```



## High-Level Overview

"""Registry for aten functions."""from __future__ import annotations__all__ = ["onnx_impl", "get_torchlib_ops"]import loggingfrom collections.abc import Callable, Sequencefrom typing import Any, TypeVarfrom typing_extensions import ParamSpecimport onnxscriptimport torchfrom torch.onnx._internal.exporter import _constants, _registration# Use ParamSpec for better type preservation instead of bound Callable TypeVar_P = ParamSpec("_P")_R = TypeVar("_R")logger = logging.getLogger("__name__")_registry: list[_registration.OnnxDecompMeta] = []def onnx_impl(    target: _registration.TorchOp | tuple[_registration.TorchOp, ...],    *,    trace_only: bool = False,    complex: bool = False,    opset_introduced: int = 18,    no_compile: bool = False,    private: bool = False,) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `onnx_impl`, `wrapper`, `get_torchlib_ops`

**Key imports**: annotations, logging, Callable, Sequence, Any, TypeVar, ParamSpec, onnxscript, torch, _constants, _registration, ops


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/exporter/_torchlib`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `logging`
- `collections.abc`: Callable, Sequence
- `typing`: Any, TypeVar
- `typing_extensions`: ParamSpec
- `onnxscript`
- `torch`
- `torch.onnx._internal.exporter`: _constants, _registration
- `torch.onnx._internal.exporter._torchlib`: ops


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/onnx/_internal/exporter/_torchlib`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tensor_typing.py_docs.md`](./_tensor_typing.py_docs.md)


## Cross-References

- **File Documentation**: `_torchlib_registry.py_docs.md`
- **Keyword Index**: `_torchlib_registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
