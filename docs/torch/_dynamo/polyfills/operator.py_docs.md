# Documentation: `torch/_dynamo/polyfills/operator.py`

## File Metadata

- **Path**: `torch/_dynamo/polyfills/operator.py`
- **Size**: 3,455 bytes (3.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Python polyfills for operator
"""

from __future__ import annotations

import operator
from typing import Any, overload, TYPE_CHECKING, TypeVar
from typing_extensions import TypeVarTuple, Unpack

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


# Most unary and binary operators are handled by BuiltinVariable (e.g., `pos`, `add`)
__all__ = ["attrgetter", "itemgetter", "methodcaller", "countOf"]


_T = TypeVar("_T")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_Ts = TypeVarTuple("_Ts")
_U = TypeVar("_U")
_U1 = TypeVar("_U1")
_U2 = TypeVar("_U2")
_Us = TypeVarTuple("_Us")


@overload
# pyrefly: ignore [inconsistent-overload]
def attrgetter(attr: str, /) -> Callable[[Any], _U]: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def attrgetter(
    attr1: str, attr2: str, /, *attrs: str
) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...


# Reference: https://docs.python.org/3/library/operator.html#operator.attrgetter
@substitute_in_graph(operator.attrgetter, is_embedded_type=True)  # type: ignore[arg-type,misc]
def attrgetter(*attrs: str) -> Callable[[Any], Any | tuple[Any, ...]]:
    if len(attrs) == 0:
        raise TypeError("attrgetter expected 1 argument, got 0")

    if any(not isinstance(attr, str) for attr in attrs):
        raise TypeError("attribute name must be a string")

    def resolve_attr(obj: Any, attr: str) -> Any:
        for name in attr.split("."):
            obj = getattr(obj, name)
        return obj

    if len(attrs) == 1:
        attr = attrs[0]

        def getter(obj: Any) -> Any:
            return resolve_attr(obj, attr)

    else:

        def getter(obj: Any) -> tuple[Any, ...]:  # type: ignore[misc]
            return tuple(resolve_attr(obj, attr) for attr in attrs)

    return getter


@overload
# pyrefly: ignore [inconsistent-overload]
def itemgetter(item: _T, /) -> Callable[[Any], _U]: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def itemgetter(
    item1: _T1, item2: _T2, /, *items: Unpack[_Ts]
) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...


# Reference: https://docs.python.org/3/library/operator.html#operator.itemgetter
@substitute_in_graph(operator.itemgetter, is_embedded_type=True)  # type: ignore[arg-type,misc]
def itemgetter(*items: Any) -> Callable[[Any], Any | tuple[Any, ...]]:
    if len(items) == 0:
        raise TypeError("itemgetter expected 1 argument, got 0")

    if len(items) == 1:
        item = items[0]

        def getter(obj: Any) -> Any:
            return obj[item]

    else:

        def getter(obj: Any) -> tuple[Any, ...]:  # type: ignore[misc]
            return tuple(obj[item] for item in items)

    return getter


# Reference: https://docs.python.org/3/library/operator.html#operator.methodcaller
@substitute_in_graph(operator.methodcaller, is_embedded_type=True)  # type: ignore[arg-type]
def methodcaller(name: str, /, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    if not isinstance(name, str):
        raise TypeError("method name must be a string")

    def caller(obj: Any) -> Any:
        return getattr(obj, name)(*args, **kwargs)

    return caller


# Reference: https://docs.python.org/3/library/operator.html#operator.countOf
@substitute_in_graph(operator.countOf, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def countOf(a: Iterable[_T], b: _T, /) -> int:
    return sum(it is b or it == b for it in a)

```



## High-Level Overview

"""Python polyfills for operator

This Python file contains 0 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `attrgetter`, `attrgetter`, `attrgetter`, `resolve_attr`, `getter`, `getter`, `itemgetter`, `itemgetter`, `itemgetter`, `getter`, `getter`, `methodcaller`, `caller`, `countOf`

**Key imports**: annotations, operator, Any, overload, TYPE_CHECKING, TypeVar, TypeVarTuple, Unpack, substitute_in_graph, Callable, Iterable


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/polyfills`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `operator`
- `typing`: Any, overload, TYPE_CHECKING, TypeVar
- `typing_extensions`: TypeVarTuple, Unpack
- `..decorators`: substitute_in_graph
- `collections.abc`: Callable, Iterable


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
- [`os.py_docs.md`](./os.py_docs.md)
- [`loader.py_docs.md`](./loader.py_docs.md)


## Cross-References

- **File Documentation**: `operator.py_docs.md`
- **Keyword Index**: `operator.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
