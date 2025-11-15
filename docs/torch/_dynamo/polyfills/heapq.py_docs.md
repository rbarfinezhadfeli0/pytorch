# Documentation: `torch/_dynamo/polyfills/heapq.py`

## File Metadata

- **Path**: `torch/_dynamo/polyfills/heapq.py`
- **Size**: 3,291 bytes (3.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Python polyfills for heapq
"""

from __future__ import annotations

import heapq
import importlib
import sys
from typing import TYPE_CHECKING, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from types import ModuleType


_T = TypeVar("_T")


# Partially copied from CPython test/support/import_helper.py
# https://github.com/python/cpython/blob/bb8791c0b75b5970d109e5557bfcca8a578a02af/Lib/test/support/import_helper.py
def _save_and_remove_modules(names: set[str]) -> dict[str, ModuleType]:
    orig_modules = {}
    prefixes = tuple(name + "." for name in names)
    for modname in list(sys.modules):
        if modname in names or modname.startswith(prefixes):
            orig_modules[modname] = sys.modules.pop(modname)
    return orig_modules


def import_fresh_module(name: str, blocked: list[str]) -> ModuleType:
    # Keep track of modules saved for later restoration as well
    # as those which just need a blocking entry removed
    names = {name, *blocked}
    orig_modules = _save_and_remove_modules(names)
    for modname in blocked:
        sys.modules[modname] = None  # type: ignore[assignment]

    try:
        return importlib.import_module(name)
    finally:
        _save_and_remove_modules(names)
        sys.modules.update(orig_modules)


# Import the pure Python heapq module, blocking the C extension
py_heapq = import_fresh_module("heapq", blocked=["_heapq"])


__all__ = [
    "_heapify_max",
    "_heappop_max",
    "_heapreplace_max",
    "heapify",
    "heappop",
    "heappush",
    "heappushpop",
    "heapreplace",
    "merge",
    "nlargest",
    "nsmallest",
]


@substitute_in_graph(heapq._heapify_max)
def _heapify_max(heap: list[_T], /) -> None:
    return py_heapq._heapify_max(heap)


@substitute_in_graph(heapq._heappop_max)  # type: ignore[attr-defined]
def _heappop_max(heap: list[_T]) -> _T:
    return py_heapq._heappop_max(heap)


@substitute_in_graph(heapq._heapreplace_max)  # type: ignore[attr-defined]
def _heapreplace_max(heap: list[_T], item: _T) -> _T:
    return py_heapq._heapreplace_max(heap, item)


@substitute_in_graph(heapq.heapify)
def heapify(heap: list[_T], /) -> None:
    return py_heapq.heapify(heap)


@substitute_in_graph(heapq.heappop)
def heappop(heap: list[_T], /) -> _T:
    return py_heapq.heappop(heap)


@substitute_in_graph(heapq.heappush)
def heappush(heap: list[_T], item: _T) -> None:
    return py_heapq.heappush(heap, item)


@substitute_in_graph(heapq.heappushpop)
def heappushpop(heap: list[_T], item: _T) -> _T:
    return py_heapq.heappushpop(heap, item)


@substitute_in_graph(heapq.heapreplace)
def heapreplace(heap: list[_T], item: _T) -> _T:
    return py_heapq.heapreplace(heap, item)


@substitute_in_graph(heapq.merge)  # type: ignore[arg-type]
def merge(*iterables, key=None, reverse=False):  # type: ignore[no-untyped-def]
    return py_heapq.merge(*iterables, key=key, reverse=reverse)


@substitute_in_graph(heapq.nlargest)  # type: ignore[arg-type]
def nlargest(n, iterable, key=None):  # type: ignore[no-untyped-def]
    return py_heapq.nlargest(n, iterable, key=key)


@substitute_in_graph(heapq.nsmallest)  # type: ignore[arg-type]
def nsmallest(n, iterable, key=None):  # type: ignore[no-untyped-def]
    return py_heapq.nsmallest(n, iterable, key=key)

```



## High-Level Overview

"""Python polyfills for heapq

This Python file contains 0 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_save_and_remove_modules`, `import_fresh_module`, `_heapify_max`, `_heappop_max`, `_heapreplace_max`, `heapify`, `heappop`, `heappush`, `heappushpop`, `heapreplace`, `merge`, `nlargest`, `nsmallest`

**Key imports**: annotations, heapq, importlib, sys, TYPE_CHECKING, TypeVar, substitute_in_graph, ModuleType


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/polyfills`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `heapq`
- `importlib`
- `sys`
- `typing`: TYPE_CHECKING, TypeVar
- `..decorators`: substitute_in_graph
- `types`: ModuleType


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

Files in the same folder (`torch/_dynamo/polyfills`):

- [`struct.py_docs.md`](./struct.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`tensor.py_docs.md`](./tensor.py_docs.md)
- [`pytree.py_docs.md`](./pytree.py_docs.md)
- [`itertools.py_docs.md`](./itertools.py_docs.md)
- [`builtins.py_docs.md`](./builtins.py_docs.md)
- [`_collections.py_docs.md`](./_collections.py_docs.md)
- [`operator.py_docs.md`](./operator.py_docs.md)
- [`os.py_docs.md`](./os.py_docs.md)
- [`loader.py_docs.md`](./loader.py_docs.md)


## Cross-References

- **File Documentation**: `heapq.py_docs.md`
- **Keyword Index**: `heapq.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
