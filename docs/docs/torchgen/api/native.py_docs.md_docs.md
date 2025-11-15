# Documentation: `docs/torchgen/api/native.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/api/native.py_docs.md`
- **Size**: 7,696 bytes (7.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/api/native.py`

## File Metadata

- **Path**: `torchgen/api/native.py`
- **Size**: 5,205 bytes (5.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import assert_never

from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import (
    ArgName,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    deviceT,
    layoutT,
    ListCType,
    MutRefCType,
    NamedCType,
    OptionalCType,
    scalarT,
    scalarTypeT,
    tensorT,
)
from torchgen.model import (
    Argument,
    FunctionSchema,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


# This file describes the translation of JIT schema to the native functions API.
# This looks a lot like the C++ API (which makes historical sense, because the
# idea was you wrote native functions to implement functions in the C++ API),
# but over time we have evolved the C++ API without actually changing our
# native:: kernels.  The intention is to make native API and dispatcher API
# line up as closely as possible, since this results in the least overhead
# (no translation is needed from dispatcher API to native API).
#
# NB: this is symint aware, you will get the non-SymInt variant for some
# dispatch entries and SymInt for others.


def name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    # TODO: delete this!
    if func.is_out_fn():
        name += "_out"
    if func.name.overload_name:
        name += f"_{func.name.overload_name}"
    return name


def argumenttype_type(
    t: Type, *, mutable: bool, binds: ArgName, symint: bool
) -> NamedCType:
    if str(t) == "Tensor?":
        tensor_type: OptionalCType = OptionalCType(BaseCType(tensorT))
        if mutable and not local.use_const_ref_for_mutable_tensors():
            return NamedCType(binds, MutRefCType(tensor_type))
        else:
            return NamedCType(binds, ConstRefCType(tensor_type))
    elif str(t) == "Tensor?[]":
        return NamedCType(
            binds, ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT))))
        )
    elif str(t) == "Scalar":
        return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
    elif str(t) == "Scalar?":
        return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
    return cpp.argumenttype_type(t, mutable=mutable, binds=binds, symint=symint)


def returns_type(rs: Sequence[Return], *, symint: bool) -> CType:
    return cpp.returns_type(rs, symint=symint)


def argument_type(a: Argument, *, binds: ArgName, symint: bool) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds, symint=symint)


def argument(
    a: Argument | SelfArgument | TensorOptionsArguments,
    *,
    is_out: bool,
    symint: bool,
) -> list[Binding]:
    # Ideally, we NEVER default native functions.  However, there are a number
    # of functions that call native:: directly and rely on the defaulting
    # existing.  So for BC, we generate defaults for non-out variants (but not
    # for out variants, where it is impossible to generate an appropriate
    # default)
    should_default = not is_out
    if isinstance(a, Argument):
        default: str | None = None
        if should_default and a.default is not None:
            default = cpp.default_expr(a.default, a.type, symint=symint)
        return [
            Binding(
                nctype=argument_type(a, binds=a.name, symint=symint),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    elif isinstance(a, SelfArgument):
        # Erase SelfArgument from the distinction
        return argument(a.argument, is_out=is_out, symint=symint)
    elif isinstance(a, TensorOptionsArguments):
        default = None
        if should_default:
            default = "{}"
        # TODO: Not sure why the arguments assigned here are for
        # TensorOptionsArguments and not the constituent pieces.  It seems
        # to matter
        return [
            Binding(
                nctype=NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT))),
                name="dtype",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("layout", OptionalCType(BaseCType(layoutT))),
                name="layout",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("device", OptionalCType(BaseCType(deviceT))),
                name="device",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("pin_memory", OptionalCType(BaseCType(boolT))),
                name="pin_memory",
                default=default,
                argument=a,
            ),
        ]
    else:
        assert_never(a)


def arguments(func: FunctionSchema, *, symint: bool) -> list[Binding]:
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    args.extend(func.arguments.non_out)
    args.extend(func.arguments.out)
    return [
        r for arg in args for r in argument(arg, symint=symint, is_out=func.is_out_fn())
    ]

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `name`, `argumenttype_type`, `returns_type`, `argument_type`, `argument`, `arguments`

**Key imports**: annotations, TYPE_CHECKING, assert_never, local, cpp, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/api`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TYPE_CHECKING
- `typing_extensions`: assert_never
- `torchgen`: local
- `torchgen.api`: cpp
- `collections.abc`: Sequence


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

Files in the same folder (`torchgen/api`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ufunc.py_docs.md`](./ufunc.py_docs.md)
- [`meta.py_docs.md`](./meta.py_docs.md)
- [`autograd.py_docs.md`](./autograd.py_docs.md)
- [`structured.py_docs.md`](./structured.py_docs.md)
- [`functionalization.py_docs.md`](./functionalization.py_docs.md)
- [`python.py_docs.md`](./python.py_docs.md)
- [`unboxing.py_docs.md`](./unboxing.py_docs.md)
- [`translate.py_docs.md`](./translate.py_docs.md)


## Cross-References

- **File Documentation**: `native.py_docs.md`
- **Keyword Index**: `native.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen/api`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torchgen/api`):

- [`meta.py_docs.md_docs.md`](./meta.py_docs.md_docs.md)
- [`translate.py_docs.md_docs.md`](./translate.py_docs.md_docs.md)
- [`unboxing.py_docs.md_docs.md`](./unboxing.py_docs.md_docs.md)
- [`ufunc.py_docs.md_docs.md`](./ufunc.py_docs.md_docs.md)
- [`cpp.py_docs.md_docs.md`](./cpp.py_docs.md_docs.md)
- [`dispatcher.py_docs.md_docs.md`](./dispatcher.py_docs.md_docs.md)
- [`autograd.py_kw.md_docs.md`](./autograd.py_kw.md_docs.md)
- [`translate.py_kw.md_docs.md`](./translate.py_kw.md_docs.md)
- [`functionalization.py_docs.md_docs.md`](./functionalization.py_docs.md_docs.md)
- [`native.py_kw.md_docs.md`](./native.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `native.py_docs.md_docs.md`
- **Keyword Index**: `native.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
