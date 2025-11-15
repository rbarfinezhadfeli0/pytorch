# Documentation: `torch/cuda/amp/autocast_mode.py`

## File Metadata

- **Path**: `torch/cuda/amp/autocast_mode.py`
- **Size**: 3,477 bytes (3.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools
import sys
from typing import Any
from typing_extensions import deprecated

import torch


__all__ = ["autocast", "custom_fwd", "custom_bwd"]


@deprecated(
    "`torch.cuda.amp.autocast(args...)` is deprecated. "
    "Please use `torch.amp.autocast('cuda', args...)` instead.",
    category=FutureWarning,
)
class autocast(torch.amp.autocast_mode.autocast):
    r"""See :class:`torch.autocast`.

    ``torch.cuda.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cuda", args...)`` instead.
    """

    # TODO: remove this conditional once we stop supporting Python < 3.13
    # Prior to Python 3.13, inspect.signature could not retrieve the correct
    # signature information for classes decorated with @deprecated (unless
    # the __new__ static method was explicitly defined);
    #
    # However, this issue has been fixed in Python 3.13 and later versions.
    if sys.version_info < (3, 13):

        def __new__(
            cls,
            enabled: bool = True,
            dtype: torch.dtype = torch.float16,
            cache_enabled: bool = True,
        ):
            return super().__new__(cls)

        def __init_subclass__(cls):
            pass

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cuda"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Preserved only for BC reasons
@deprecated(
    "`torch.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated. "
    "Please use `torch.amp.autocast_mode._cast(value, 'cuda', dtype)` instead.",
    category=FutureWarning,
)
def _cast(value, dtype):
    return torch.amp.autocast_mode._cast(value, "cuda", dtype)


@deprecated(
    "`torch.cuda.amp.custom_fwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``torch.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(torch.amp.custom_fwd, device_type="cuda")(
        fwd=fwd, cast_inputs=cast_inputs
    )


@deprecated(
    "`torch.cuda.amp.custom_bwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    """
    ``torch.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(torch.amp.custom_bwd, device_type="cuda")(bwd)

```



## High-Level Overview

r"""See :class:`torch.autocast`.    ``torch.cuda.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cuda", args...)`` instead.

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `autocast`

**Functions defined**: `__new__`, `__init_subclass__`, `__init__`, `__enter__`, `__exit__`, `__call__`, `_cast`, `custom_fwd`, `custom_bwd`

**Key imports**: functools, sys, Any, deprecated, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/cuda/amp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `sys`
- `typing`: Any
- `typing_extensions`: deprecated
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/cuda/amp`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`grad_scaler.py_docs.md`](./grad_scaler.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)


## Cross-References

- **File Documentation**: `autocast_mode.py_docs.md`
- **Keyword Index**: `autocast_mode.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
