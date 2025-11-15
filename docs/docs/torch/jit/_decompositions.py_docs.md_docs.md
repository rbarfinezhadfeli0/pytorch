# Documentation: `docs/torch/jit/_decompositions.py_docs.md`

## File Metadata

- **Path**: `docs/torch/jit/_decompositions.py_docs.md`
- **Size**: 7,102 bytes (6.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/jit/_decompositions.py`

## File Metadata

- **Path**: `torch/jit/_decompositions.py`
- **Size**: 4,507 bytes (4.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
from torch import Tensor


aten = torch.ops.aten
import inspect
import warnings
from collections.abc import Callable
from typing import Optional, TypeVar
from typing_extensions import ParamSpec

from torch.types import Number


decomposition_table: dict[str, torch.jit.ScriptFunction] = {}
function_name_set: set[str] = set()

_T = TypeVar("_T")
_P = ParamSpec("_P")


def check_decomposition_has_type_annotations(f) -> None:
    inspect_empty = inspect._empty  # type: ignore[attr-defined]
    sig = inspect.signature(f)
    for param in sig.parameters.values():
        assert param.annotation != inspect_empty, (
            f"No signature on param {param.name} for function {f.name}"
        )

    assert sig.return_annotation != inspect_empty, (
        f"No return annotation for function {f.name}"
    )


def signatures_match(decomposition_sig, torch_op_sig):
    decomp_params = decomposition_sig.parameters
    op_params = torch_op_sig.parameters

    if len(decomp_params) != len(op_params):
        return False

    for decomp_param, op_param in zip(decomp_params.values(), op_params.values()):
        # can't check full equality yet because not all fields are correctly deduced
        # in the torch_op_sig - like default value
        # can't check 'kind' bc
        # kwarg-only values with defaults not yet supported in TS
        inspect_empty = inspect._empty  # type: ignore[attr-defined]
        for field in ["name", "annotation"]:
            if field == "name" and decomp_param.name == "self":
                warnings.warn(
                    "PyTorch uses 'input' instead of 'self' on public api", stacklevel=2
                )

            if getattr(decomp_param, field) != getattr(op_param, field):
                return False

        decomp_default = decomp_param.default
        op_default = op_param.default
        # default value not always correctly inferred as being present on torch schema,
        # but if specified on both they should be equal
        if decomp_default != inspect_empty and op_default != inspect_empty:
            if decomp_default != op_default:
                return False

    return decomposition_sig.return_annotation == torch_op_sig.return_annotation


def register_decomposition(
    aten_op: torch._ops.OpOverload,
    registry: Optional[dict[str, torch.jit.ScriptFunction]] = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decomposition_decorator(f: Callable[_P, _T]) -> Callable[_P, _T]:
        nonlocal registry
        if registry is None:
            registry = decomposition_table

        assert isinstance(aten_op, torch._ops.OpOverload)

        # Need unique name for jit function serialization
        assert f.__name__ not in function_name_set, (
            f"Duplicated function name {f.__name__}"
        )
        function_name_set.add(f.__name__)

        scripted_func = torch.jit.script(f)
        torch._C._jit_pass_inline(scripted_func.graph)

        for _ in range(2):
            torch._C._jit_pass_peephole(scripted_func.graph)
            torch._C._jit_pass_constant_propagation(scripted_func.graph)

        registry[str(aten_op._schema)] = scripted_func
        return f

    return decomposition_decorator


# TODO: replace torch.sigmoid -> aten.sigmoid


@register_decomposition(aten.var.correction)
def var_decomposition(
    input: Tensor,
    dim: Optional[list[int]] = None,
    correction: Optional[Number] = None,
    keepdim: bool = False,
) -> Tensor:
    if dim is None:
        dim_i: list[int] = []
        dim = dim_i

    if isinstance(dim, (tuple, list)) and len(dim) == 0:
        n = input.numel()
    else:
        n = 1
        for dim_i in dim:  # type: ignore[assignment]
            n *= input.shape[dim_i]  # type: ignore[call-overload]

    mean = aten.mean(input, dim, True)
    sub = input - mean
    sq = sub * sub
    sum = aten.sum(sq, dim, keepdim)

    if correction is None:
        denom = float(n - 1)
    else:
        if isinstance(correction, int):
            denom = float(n - correction)
        elif isinstance(correction, float):
            denom = float(n) - correction
        else:
            raise RuntimeError("correction must be int or float")

    # pyrefly: ignore [no-matching-overload]
    return sum / max(0, denom)


@register_decomposition(aten.var.default)
def var(input: Tensor, unbiased: bool = True) -> Tensor:
    return var_decomposition(input, correction=(1 if unbiased else 0))

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `check_decomposition_has_type_annotations`, `signatures_match`, `register_decomposition`, `decomposition_decorator`, `var_decomposition`, `var`

**Key imports**: torch, Tensor, inspect, warnings, Callable, Optional, TypeVar, ParamSpec, Number


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `inspect`
- `warnings`
- `collections.abc`: Callable
- `typing`: Optional, TypeVar
- `typing_extensions`: ParamSpec
- `torch.types`: Number


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

Files in the same folder (`torch/jit`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_dataclass_impls.py_docs.md`](./_dataclass_impls.py_docs.md)
- [`quantized.py_docs.md`](./quantized.py_docs.md)
- [`frontend.py_docs.md`](./frontend.py_docs.md)
- [`_builtins.py_docs.md`](./_builtins.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)
- [`_state.py_docs.md`](./_state.py_docs.md)
- [`_await.py_docs.md`](./_await.py_docs.md)


## Cross-References

- **File Documentation**: `_decompositions.py_docs.md`
- **Keyword Index**: `_decompositions.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/jit`):

- [`_check.py_kw.md_docs.md`](./_check.py_kw.md_docs.md)
- [`_shape_functions.py_docs.md_docs.md`](./_shape_functions.py_docs.md_docs.md)
- [`_trace.py_kw.md_docs.md`](./_trace.py_kw.md_docs.md)
- [`_logging.py_docs.md_docs.md`](./_logging.py_docs.md_docs.md)
- [`_async.py_kw.md_docs.md`](./_async.py_kw.md_docs.md)
- [`_state.py_docs.md_docs.md`](./_state.py_docs.md_docs.md)
- [`_decomposition_utils.py_kw.md_docs.md`](./_decomposition_utils.py_kw.md_docs.md)
- [`frontend.py_docs.md_docs.md`](./frontend.py_docs.md_docs.md)
- [`_check.py_docs.md_docs.md`](./_check.py_docs.md_docs.md)
- [`_script.pyi_docs.md_docs.md`](./_script.pyi_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_decompositions.py_docs.md_docs.md`
- **Keyword Index**: `_decompositions.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
