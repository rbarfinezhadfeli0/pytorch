# Documentation: `docs/torchgen/context.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/context.py_docs.md`
- **Size**: 6,850 bytes (6.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/context.py`

## File Metadata

- **Path**: `torchgen/context.py`
- **Size**: 4,056 bytes (3.96 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import contextlib
import functools
from typing import Any, Optional, TYPE_CHECKING, TypeVar, Union

import torchgen.local as local
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
)
from torchgen.utils import context, S, T


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


# Helper functions for defining generators on things in the model

F = TypeVar(
    "F",
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    Union[NativeFunction, NativeFunctionsGroup],
    Union[NativeFunction, NativeFunctionsViewGroup],
)

F2 = TypeVar(
    "F2",
    NativeFunction,
    NativeFunctionsGroup,
    Optional[NativeFunction],
    bool,
    str,
)

F3 = TypeVar("F3", tuple[NativeFunction, Any], list[NativeFunction])


@contextlib.contextmanager
def native_function_manager(
    g: NativeFunctionsGroup | NativeFunctionsViewGroup | NativeFunction,
) -> Iterator[None]:
    if isinstance(g, NativeFunctionsGroup):
        # By default, we associate all errors with structured native functions
        # with the out variant.  In some cases, it might be better to have
        # a more specific place to hang things; if so, use
        # native_function_manager again on the inside
        f = g.out
    elif isinstance(g, NativeFunctionsViewGroup):
        # We associate errors with the view operator
        f = g.view
    else:
        f = g
    with context(lambda: f"in native_functions.yaml line {f.loc}:\n  {f.func}"):
        with local.parametrize(
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=f.part_of_structured_group,
        ):
            yield


# Given a function that operates on NativeFunction, wrap it into a new function
# that sets some appropriate context managers for that native function.
# YOU MUST WRAP FUNCTIONS IN THIS for calls to api modules to be sound
# (you will get an error if we try to access the local variables without having
# set them).
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:
    @functools.wraps(func)
    def wrapper(f: F) -> T:
        with native_function_manager(f):
            return func(f)

    return wrapper


def with_native_function_and(func: Callable[[F, F2], T]) -> Callable[[F, F2], T]:
    @functools.wraps(func)
    def wrapper(f: F, f2: F2) -> T:
        # The first native_function is assumed to be the one with the appropriate context.
        with native_function_manager(f):
            return func(f, f2)

    return wrapper


def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        with native_function_manager(f):
            return func(slf, f)

    return wrapper


def method_with_nested_native_function(
    func: Callable[[S, F3], T],
) -> Callable[[S, F3], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F3) -> T:
        with native_function_manager(f[0]):
            return func(slf, f)

    return wrapper


# Convenience decorator for functions that explicitly take in a BackendIndex,
# instead of indirectly taking one in as a closure
def with_native_function_and_index(
    func: Callable[[F, BackendIndex], T],
) -> Callable[[F, BackendIndex], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_index: BackendIndex) -> T:
        with native_function_manager(f):
            return func(f, backend_index)

    return wrapper


# Convenience decorator for functions that explicitly take in a Dict of BackendIndices
def with_native_function_and_indices(
    func: Callable[[F, dict[DispatchKey, BackendIndex]], T],
) -> Callable[[F, dict[DispatchKey, BackendIndex]], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_indices: dict[DispatchKey, BackendIndex]) -> T:
        with native_function_manager(f):
            return func(f, backend_indices)

    return wrapper

```



## High-Level Overview


This Python file contains 0 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `native_function_manager`, `with_native_function`, `wrapper`, `with_native_function_and`, `wrapper`, `method_with_native_function`, `wrapper`, `method_with_nested_native_function`, `wrapper`, `with_native_function_and_index`, `wrapper`, `with_native_function_and_indices`, `wrapper`

**Key imports**: annotations, contextlib, functools, Any, Optional, TYPE_CHECKING, TypeVar, Union, torchgen.local as local, context, S, T, Callable, Iterator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `contextlib`
- `functools`
- `typing`: Any, Optional, TYPE_CHECKING, TypeVar, Union
- `torchgen.local as local`
- `torchgen.utils`: context, S, T
- `collections.abc`: Callable, Iterator


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

Files in the same folder (`torchgen`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`gen_backend_stubs.py_docs.md`](./gen_backend_stubs.py_docs.md)
- [`local.py_docs.md`](./local.py_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`model.py_docs.md`](./model.py_docs.md)
- [`yaml_utils.py_docs.md`](./yaml_utils.py_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`gen_schema_utils.py_docs.md`](./gen_schema_utils.py_docs.md)
- [`gen.py_docs.md`](./gen.py_docs.md)


## Cross-References

- **File Documentation**: `context.py_docs.md`
- **Keyword Index**: `context.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torchgen`):

- [`gen_functionalization_type.py_docs.md_docs.md`](./gen_functionalization_type.py_docs.md_docs.md)
- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`context.py_kw.md_docs.md`](./context.py_kw.md_docs.md)
- [`native_function_generation.py_kw.md_docs.md`](./native_function_generation.py_kw.md_docs.md)
- [`gen_schema_utils.py_docs.md_docs.md`](./gen_schema_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`gen_aoti_c_shim.py_docs.md_docs.md`](./gen_aoti_c_shim.py_docs.md_docs.md)
- [`local.py_docs.md_docs.md`](./local.py_docs.md_docs.md)
- [`gen.py_kw.md_docs.md`](./gen.py_kw.md_docs.md)
- [`gen_aoti_c_shim.py_kw.md_docs.md`](./gen_aoti_c_shim.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `context.py_docs.md_docs.md`
- **Keyword Index**: `context.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
