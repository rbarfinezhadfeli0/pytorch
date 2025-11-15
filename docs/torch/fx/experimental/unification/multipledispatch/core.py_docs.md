# Documentation: `torch/fx/experimental/unification/multipledispatch/core.py`

## File Metadata

- **Path**: `torch/fx/experimental/unification/multipledispatch/core.py`
- **Size**: 2,859 bytes (2.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import inspect
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import TypeVarTuple, Unpack

from .dispatcher import Dispatcher, MethodDispatcher


global_namespace = {}  # type: ignore[var-annotated]

__all__ = ["dispatch", "ismethod"]

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


def dispatch(
    *types: Unpack[Ts], **kwargs: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Dispatch function on the types of the inputs
    Supports dispatch on all non-keyword arguments.
    Collects implementations based on the function name.  Ignores namespaces.
    If ambiguous type signatures occur a warning is raised when the function is
    defined suggesting the additional method to break the ambiguity.

    Example:
        >>> # xdoctest: +SKIP
        >>> @dispatch(int)
        ... def f(x):
        ...     return x + 1
        >>> @dispatch(float)
        ... def f(x):
        ...     return x - 1
        >>> # xdoctest: +SKIP
        >>> f(3)
        4
        >>> f(3.0)
        2.0
        >>> # Specify an isolated namespace with the namespace keyword argument
        >>> my_namespace = {}
        >>> @dispatch(int, namespace=my_namespace)
        ... def foo(x):
        ...     return x + 1
        >>> # Dispatch on instance methods within classes
        >>> class MyClass(object):
        ...     @dispatch(list)
        ...     def __init__(self, data):
        ...         self.data = data
        ...
        ...     @dispatch(int)
        ...     def __init__(self, datum):
        ...         self.data = [datum]
        >>> MyClass([1, 2, 3]).data
        [1, 2, 3]
        >>> MyClass(3).data
        [3]
    """
    namespace = kwargs.get("namespace", global_namespace)

    types_tuple: tuple[type, ...] = tuple(types)  # type: ignore[arg-type]

    def _df(func):
        name = func.__name__

        if ismethod(func):
            dispatcher = inspect.currentframe().f_back.f_locals.get(  # type: ignore[union-attr]
                name,  # type: ignore[union-attr]
                MethodDispatcher(name),
            )
        else:
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        dispatcher.add(types_tuple, func)
        return dispatcher

    return _df


def ismethod(func):
    """Is func a method?
    Note that this has to work as the method is defined but before the class is
    defined.  At this stage methods look like functions.
    """
    if hasattr(inspect, "signature"):
        signature = inspect.signature(func)
        return signature.parameters.get("self", None) is not None
    else:
        spec = inspect.getfullargspec(func)  # type: ignore[union-attr, assignment]
        return spec and spec.args and spec.args[0] == "self"

```



## High-Level Overview

"""Dispatch function on the types of the inputs    Supports dispatch on all non-keyword arguments.    Collects implementations based on the function name.  Ignores namespaces.    If ambiguous type signatures occur a warning is raised when the function is    defined suggesting the additional method to break the ambiguity.    Example:        >>> # xdoctest: +SKIP        >>> @dispatch(int)        ... def f(x):        ...     return x + 1        >>> @dispatch(float)        ... def f(x):        ...     return x - 1        >>> # xdoctest: +SKIP        >>> f(3)        4        >>> f(3.0)        2.0        >>> # Specify an isolated namespace with the namespace keyword argument        >>> my_namespace = {}        >>> @dispatch(int, namespace=my_namespace)        ... def foo(x):        ...     return x + 1        >>> # Dispatch on instance methods within classes        >>> class MyClass(object):        ...     @dispatch(list)        ...     def __init__(self, data):        ...         self.data = data        ...

This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyClass`

**Functions defined**: `dispatch`, `f`, `f`, `foo`, `__init__`, `__init__`, `_df`, `ismethod`

**Key imports**: inspect, Callable, Any, TypeVar, TypeVarTuple, Unpack, Dispatcher, MethodDispatcher


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental/unification/multipledispatch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `inspect`
- `collections.abc`: Callable
- `typing`: Any, TypeVar
- `typing_extensions`: TypeVarTuple, Unpack
- `.dispatcher`: Dispatcher, MethodDispatcher


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

Files in the same folder (`torch/fx/experimental/unification/multipledispatch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`conflict.py_docs.md`](./conflict.py_docs.md)
- [`dispatcher.py_docs.md`](./dispatcher.py_docs.md)
- [`variadic.py_docs.md`](./variadic.py_docs.md)


## Cross-References

- **File Documentation**: `core.py_docs.md`
- **Keyword Index**: `core.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
