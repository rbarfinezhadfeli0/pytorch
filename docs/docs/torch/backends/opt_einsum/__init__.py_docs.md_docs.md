# Documentation: `docs/torch/backends/opt_einsum/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/backends/opt_einsum/__init__.py_docs.md`
- **Size**: 6,908 bytes (6.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/backends/opt_einsum/__init__.py`

## File Metadata

- **Path**: `torch/backends/opt_einsum/__init__.py`
- **Size**: 3,904 bytes (3.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache as _lru_cache
from typing import Any

from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


try:
    import opt_einsum as _opt_einsum  # type: ignore[import]
except ImportError:
    _opt_einsum = None


@_lru_cache
def is_available() -> bool:
    r"""Return a bool indicating if opt_einsum is currently available.

    You must install opt-einsum in order for torch to automatically optimize einsum. To
    make opt-einsum available, you can install it along with torch: ``pip install torch[opt-einsum]``
    or by itself: ``pip install opt-einsum``. If the package is installed, torch will import
    it automatically and use it accordingly. Use this function to check whether opt-einsum
    was installed and properly imported by torch.
    """
    return _opt_einsum is not None


def get_opt_einsum() -> Any:
    r"""Return the opt_einsum package if opt_einsum is currently available, else None."""
    return _opt_einsum


def _set_enabled(_enabled: bool) -> None:
    if not is_available() and _enabled:
        raise ValueError(
            f"opt_einsum is not available, so setting `enabled` to {_enabled} will not reap "
            "the benefits of calculating an optimal path for einsum. torch.einsum will "
            "fall back to contracting from left to right. To enable this optimal path "
            "calculation, please install opt-einsum."
        )
    global enabled
    enabled = _enabled


def _get_enabled() -> bool:
    return enabled


def _set_strategy(_strategy: str) -> None:
    if not is_available():
        raise ValueError(
            f"opt_einsum is not available, so setting `strategy` to {_strategy} will not be meaningful. "
            "torch.einsum will bypass path calculation and simply contract from left to right. "
            "Please install opt_einsum or unset `strategy`."
        )
    if not enabled:
        raise ValueError(
            f"opt_einsum is not enabled, so setting a `strategy` to {_strategy} will not be meaningful. "
            "torch.einsum will bypass path calculation and simply contract from left to right. "
            "Please set `enabled` to `True` as well or unset `strategy`."
        )
    if _strategy not in ["auto", "greedy", "optimal"]:
        raise ValueError(
            f"`strategy` must be one of the following: [auto, greedy, optimal] but is {_strategy}"
        )
    global strategy
    strategy = _strategy


def _get_strategy() -> str:
    # pyrefly: ignore [bad-return]
    return strategy


def set_flags(_enabled=None, _strategy=None):
    orig_flags = (enabled, None if not is_available() else strategy)
    if _enabled is not None:
        _set_enabled(_enabled)
    if _strategy is not None:
        _set_strategy(_strategy)
    return orig_flags


@contextmanager
def flags(enabled=None, strategy=None):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, strategy)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.opt_einsum.enabled = True


class OptEinsumModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    global enabled
    enabled = ContextProp(_get_enabled, _set_enabled)
    global strategy
    strategy = None
    if is_available():
        strategy = ContextProp(_get_strategy, _set_strategy)


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = OptEinsumModule(sys.modules[__name__], __name__)

enabled = bool(is_available())
strategy = "auto" if is_available() else None

```



## High-Level Overview

r"""Return a bool indicating if opt_einsum is currently available.    You must install opt-einsum in order for torch to automatically optimize einsum. To    make opt-einsum available, you can install it along with torch: ``pip install torch[opt-einsum]``    or by itself: ``pip install opt-einsum``. If the package is installed, torch will import    it automatically and use it accordingly. Use this function to check whether opt-einsum    was installed and properly imported by torch.

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OptEinsumModule`

**Functions defined**: `is_available`, `get_opt_einsum`, `_set_enabled`, `_get_enabled`, `_set_strategy`, `_get_strategy`, `set_flags`, `flags`, `__init__`

**Key imports**: sys, warnings, contextmanager, lru_cache as _lru_cache, Any, __allow_nonbracketed_mutation, ContextProp, PropModule, opt_einsum as _opt_einsum  , it automatically and use it accordingly. Use this function to check whether opt


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/backends/opt_einsum`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `warnings`
- `contextlib`: contextmanager
- `functools`: lru_cache as _lru_cache
- `typing`: Any
- `torch.backends`: __allow_nonbracketed_mutation, ContextProp, PropModule
- `opt_einsum as _opt_einsum  `
- `it automatically and use it accordingly. Use this function to check whether opt`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/backends/opt_einsum`):



## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/backends/opt_einsum`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/backends/opt_einsum`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/backends/opt_einsum`):

- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
