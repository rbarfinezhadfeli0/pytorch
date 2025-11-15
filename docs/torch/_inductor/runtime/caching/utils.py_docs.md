# Documentation: `torch/_inductor/runtime/caching/utils.py`

## File Metadata

- **Path**: `torch/_inductor/runtime/caching/utils.py`
- **Size**: 3,432 bytes (3.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Utility functions for caching operations in PyTorch Inductor runtime.

This module provides helper functions for pickling/unpickling operations
with error handling, LRU caching decorators, and type-safe serialization
utilities used throughout the caching system.
"""

import pickle
from collections.abc import Callable
from functools import lru_cache, partial, wraps
from typing import Any
from typing_extensions import ParamSpec, TypeVar

from . import exceptions


# Type specification for function parameters
P = ParamSpec("P")
# Type variable for function return values
R = TypeVar("R")


def _lru_cache(fn: Callable[P, R]) -> Callable[P, R]:
    """LRU cache decorator with TypeError fallback.

    Provides LRU caching with a fallback mechanism that calls the original
    function if caching fails due to unhashable arguments. Uses a cache
    size of 64 with typed comparison.

    Args:
        fn: The function to be cached.

    Returns:
        A wrapper function that attempts caching with fallback to original function.
    """
    cached_fn = lru_cache(maxsize=64, typed=True)(fn)

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[type-var]
        try:
            return cached_fn(*args, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            return fn(*args, **kwargs)

    return wrapper


@_lru_cache
def _try_pickle(to_pickle: Any, raise_if_failed: type = exceptions.CacheError) -> bytes:
    """Attempt to pickle an object with error handling.

    Tries to serialize an object using pickle.dumps with appropriate error
    handling and custom exception raising.

    Args:
        to_pickle: The object to be pickled.
        raise_if_failed: Exception class to raise if pickling fails.

    Returns:
        The pickled bytes representation of the object.

    Raises:
        The exception class specified in raise_if_failed if pickling fails.
    """
    try:
        pickled: bytes = pickle.dumps(to_pickle)
    except (pickle.PicklingError, AttributeError) as err:
        raise raise_if_failed(to_pickle) from err
    return pickled


# Specialized pickle function for cache keys with KeyPicklingError handling.
_try_pickle_key: Callable[[Any], bytes] = partial(
    _try_pickle, raise_if_failed=exceptions.KeyPicklingError
)
# Specialized pickle function for cache values with ValuePicklingError handling.
_try_pickle_value: Callable[[Any], bytes] = partial(
    _try_pickle, raise_if_failed=exceptions.ValuePicklingError
)


@_lru_cache
def _try_unpickle(pickled: bytes, raise_if_failed: type = exceptions.CacheError) -> Any:
    """Attempt to unpickle bytes with error handling.

    Tries to deserialize bytes using pickle.loads with appropriate error
    handling and custom exception raising.

    Args:
        pickled: The bytes to be unpickled.
        raise_if_failed: Exception class to raise if unpickling fails.

    Returns:
        The unpickled object.

    Raises:
        The exception class specified in raise_if_failed if unpickling fails.
    """
    try:
        unpickled: Any = pickle.loads(pickled)
    except pickle.UnpicklingError as err:
        raise raise_if_failed(pickled) from err
    return unpickled


# Specialized unpickle function for cache keys with KeyUnPicklingError handling.
_try_unpickle_value: Callable[[Any], bytes] = partial(
    _try_unpickle, raise_if_failed=exceptions.ValueUnPicklingError
)

```



## High-Level Overview

"""Utility functions for caching operations in PyTorch Inductor runtime.This module provides helper functions for pickling/unpickling operationswith error handling, LRU caching decorators, and type-safe serializationutilities used throughout the caching system.

This Python file contains 4 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_lru_cache`, `wrapper`, `_try_pickle`, `_try_unpickle`

**Key imports**: pickle, Callable, lru_cache, partial, wraps, Any, ParamSpec, TypeVar, exceptions


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/runtime/caching`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `pickle`
- `collections.abc`: Callable
- `functools`: lru_cache, partial, wraps
- `typing`: Any
- `typing_extensions`: ParamSpec, TypeVar
- `.`: exceptions


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor/runtime/caching`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`exceptions.py_docs.md`](./exceptions.py_docs.md)
- [`implementations.py_docs.md`](./implementations.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`interfaces.py_docs.md`](./interfaces.py_docs.md)
- [`locks.py_docs.md`](./locks.py_docs.md)
- [`context.py_docs.md`](./context.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
