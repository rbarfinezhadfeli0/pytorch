# Documentation: `torch/multiprocessing/__init__.py`

## File Metadata

- **Path**: `torch/multiprocessing/__init__.py`
- **Size**: 3,456 bytes (3.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
"""torch.multiprocessing is a wrapper around the native :mod:`multiprocessing` module.

It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""

import multiprocessing
import sys

import torch

from .reductions import init_reductions


__all__ = ["set_sharing_strategy", "get_sharing_strategy", "get_all_sharing_strategies"]


from multiprocessing import *  # noqa: F403


__all__ += multiprocessing.__all__  # noqa: PLE0605 type: ignore[attr-defined]


# This call adds a Linux specific prctl(2) wrapper function to this module.
# See https://github.com/pytorch/pytorch/pull/14391 for more information.
torch._C._multiprocessing_init()


"""Add helper function to spawn N processes and wait for completion of any of
them. This depends `mp.get_context` which was added in Python 3.4."""
from .spawn import (
    ENV_VAR_PARALLEL_START,
    ProcessContext,
    ProcessExitedException,
    ProcessRaisedException,
    spawn,
    SpawnContext,
    start_processes,
)


if sys.platform == "darwin" or sys.platform == "win32":
    _sharing_strategy = "file_system"
    _all_sharing_strategies = {"file_system"}
else:
    _sharing_strategy = "file_descriptor"
    _all_sharing_strategies = {"file_descriptor", "file_system"}


def set_sharing_strategy(new_strategy):
    """Set the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    assert new_strategy in _all_sharing_strategies
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    """Return the current strategy for sharing CPU tensors."""
    return _sharing_strategy


def get_all_sharing_strategies():
    """Return a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies


def _set_thread_name(name: str) -> None:
    """Set the name of the current thread.

    Args:
        name (str): Name of the current thread.
    """
    torch._C._set_thread_name(name)


def _get_thread_name() -> str:
    """Get the name of the current thread.

    Returns:
        str: Name of the current thread.
    """
    return torch._C._get_thread_name()


init_reductions()

# Leak ResourceTracker at exit for Python-3.12 on MacOS
# See https://github.com/pytorch/pytorch/issues/153050 and
# https://github.com/python/cpython/issues/88887 for more details
from multiprocessing.resource_tracker import ResourceTracker as _RT


if (
    sys.platform == "darwin"
    and sys.version_info >= (3, 12, 2)
    and hasattr(_RT, "__del__")
):
    import atexit

    def _leak_RT_at_exit():
        def _noop(x):
            pass

        _RT.__del__ = _noop  # type: ignore[attr-defined]

    atexit.register(_leak_RT_at_exit)

```



## High-Level Overview

"""torch.multiprocessing is a wrapper around the native :mod:`multiprocessing` module.It registers custom reducers, that use shared memory to provide sharedviews on the same data in different processes. Once the tensor/storage is movedto shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possibleto send it to other processes without making any copies.The API is 100% compatible with the original module - it's enough to change``import multiprocessing`` to ``import torch.multiprocessing`` to have all thetensors sent through the queues or shared via other mechanisms, moved to sharedmemory.Because of the similarity of APIs we do not document most of this packagecontents, and we recommend referring to very good docs of the original module.

This Python file contains 0 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `set_sharing_strategy`, `get_sharing_strategy`, `get_all_sharing_strategies`, `_set_thread_name`, `_get_thread_name`, `_leak_RT_at_exit`, `_noop`

**Key imports**: multiprocessing, torch.multiprocessing, multiprocessing, sys, torch, init_reductions, ResourceTracker as _RT, atexit


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/multiprocessing`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `multiprocessing`
- `torch.multiprocessing`
- `sys`
- `torch`
- `.reductions`: init_reductions
- `multiprocessing.resource_tracker`: ResourceTracker as _RT
- `atexit`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/multiprocessing`):

- [`pool.py_docs.md`](./pool.py_docs.md)
- [`_atfork.py_docs.md`](./_atfork.py_docs.md)
- [`spawn.py_docs.md`](./spawn.py_docs.md)
- [`cuda_multiprocessing.md_docs.md`](./cuda_multiprocessing.md_docs.md)
- [`reductions.py_docs.md`](./reductions.py_docs.md)
- [`queue.py_docs.md`](./queue.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
