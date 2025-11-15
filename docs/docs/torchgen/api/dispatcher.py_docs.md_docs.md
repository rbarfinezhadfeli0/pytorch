# Documentation: `docs/torchgen/api/dispatcher.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/api/dispatcher.py_docs.md`
- **Size**: 6,199 bytes (6.05 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/api/dispatcher.py`

## File Metadata

- **Path**: `torchgen/api/dispatcher.py`
- **Size**: 3,479 bytes (3.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from typing_extensions import assert_never

from torchgen.api import cpp
from torchgen.api.types import ArgName, Binding, CType, NamedCType
from torchgen.model import (
    Argument,
    FunctionSchema,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import concatMap


if TYPE_CHECKING:
    from collections.abc import Sequence


# This file describes the translation of JIT schema to the dispatcher
# API, the *unboxed* calling convention by which invocations through
# the dispatcher are made.  Historically, the dispatcher API matched
# the C++ API, but with the establishment of the boxed API, we've
# made changes to the dispatcher API to so that the unboxed API
# better aligns with the boxed API.  The dispatcher API hooks heavily
# into our template based boxing/unboxing machinery, so changes
# to this convention will usually need template updates too.
#
# Prominent characteristics of the dispatcher API:
#
#   - dtype, layout, device and pin_memory are represented as separate
#     arguments.
#


def name(func: FunctionSchema) -> str:
    return cpp.name(func)


def argumenttype_type(
    t: Type,
    *,
    mutable: bool,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = True,
) -> NamedCType:
    # This is a faux amis.  If it makes sense in the future to add
    # more special cases here, or invert things so cpp.argument_type
    # calls this, or just completely inline the function, please do
    # it.
    return cpp.argumenttype_type(
        t,
        mutable=mutable,
        binds=binds,
        symint=symint,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
    )


def argument_type(
    a: Argument,
    *,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = True,
) -> NamedCType:
    return argumenttype_type(
        a.type,
        mutable=a.is_write,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
        symint=symint,
    )


def returns_type(rs: Sequence[Return], *, symint: bool = True) -> CType:
    # At present, there is no difference. But there could be!
    return cpp.returns_type(rs, symint=symint)


def jit_arguments(func: FunctionSchema) -> list[Argument]:
    def to_argument(
        a: Argument | TensorOptionsArguments | SelfArgument,
    ) -> list[Argument]:
        if isinstance(a, Argument):
            return [a]
        elif isinstance(a, SelfArgument):
            return [a.argument]
        elif isinstance(a, TensorOptionsArguments):
            return [a.dtype, a.layout, a.device, a.pin_memory]
        else:
            assert_never(a)

    return list(
        concatMap(
            to_argument,
            itertools.chain(
                func.arguments.positional, func.arguments.kwarg_only, func.arguments.out
            ),
        )
    )


def argument(
    a: Argument, *, remove_non_owning_ref_types: bool = False, symint: bool = True
) -> Binding:
    return Binding(
        nctype=argument_type(
            a,
            binds=a.name,
            remove_non_owning_ref_types=remove_non_owning_ref_types,
            symint=symint,
        ),
        name=a.name,
        argument=a,
    )


def arguments(func: FunctionSchema, *, symint: bool = True) -> list[Binding]:
    return [argument(a, symint=symint) for a in jit_arguments(func)]

```



## High-Level Overview


This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `name`, `argumenttype_type`, `argument_type`, `returns_type`, `jit_arguments`, `to_argument`, `argument`, `arguments`

**Key imports**: annotations, itertools, TYPE_CHECKING, assert_never, cpp, ArgName, Binding, CType, NamedCType, concatMap, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/api`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `itertools`
- `typing`: TYPE_CHECKING
- `typing_extensions`: assert_never
- `torchgen.api`: cpp
- `torchgen.api.types`: ArgName, Binding, CType, NamedCType
- `torchgen.utils`: concatMap
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
- [`native.py_docs.md`](./native.py_docs.md)


## Cross-References

- **File Documentation**: `dispatcher.py_docs.md`
- **Keyword Index**: `dispatcher.py_kw.md`
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
- [`autograd.py_kw.md_docs.md`](./autograd.py_kw.md_docs.md)
- [`translate.py_kw.md_docs.md`](./translate.py_kw.md_docs.md)
- [`functionalization.py_docs.md_docs.md`](./functionalization.py_docs.md_docs.md)
- [`native.py_kw.md_docs.md`](./native.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `dispatcher.py_docs.md_docs.md`
- **Keyword Index**: `dispatcher.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
