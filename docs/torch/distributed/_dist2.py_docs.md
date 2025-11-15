# Documentation: `torch/distributed/_dist2.py`

## File Metadata

- **Path**: `torch/distributed/_dist2.py`
- **Size**: 4,848 bytes (4.73 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
This is an experimental new API for PyTorch Distributed. This is actively in development and subject to change or deletion entirely.

This is intended as a proving ground for more flexible and object oriented distributed APIs.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from typing import Protocol, Union

import torch
from torch._C._distributed_c10d import (
    _current_process_group,
    _set_process_group,
    ProcessGroup,
    ReduceOp,
    Store,
)
from torch.distributed.rendezvous import rendezvous


_BACKENDS: dict[str, "ProcessGroupFactory"] = {}

__all__ = [
    "ProcessGroup",
    "ReduceOp",
    "ProcessGroupFactory",
    "register_backend",
    "new_group",
    "current_process_group",
    "process_group",
]


class ProcessGroupFactory(Protocol):
    """Protocol for process group factories."""

    def __call__(
        self,
        store: Store,
        rank: int,
        world_size: int,
        timeout: timedelta,
        device: torch.device,
        **kwargs: object,
    ) -> ProcessGroup: ...


def register_backend(name: str, func: ProcessGroupFactory) -> None:
    """
    Register a new process group backend.

    Args:
        name: The name of the backend.
        func: The function to create the process group.
    """
    if name in _BACKENDS:
        raise ValueError(f"Backend {name} already registered")

    _BACKENDS[name] = func


def _gloo_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device,
    **kwargs: object,
) -> ProcessGroup:
    from torch.distributed import ProcessGroupGloo

    if len(kwargs) != 0:
        raise AssertionError("Gloo backend received unexpected kwargs")

    backend_class = ProcessGroupGloo(store, rank, world_size, timeout)
    backend_class._set_sequence_number_for_group()

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    # register devices
    pg._register_backend(device, ProcessGroup.BackendType.GLOO, backend_class)
    pg._register_backend(
        torch.device("cpu"), ProcessGroup.BackendType.GLOO, backend_class
    )
    if torch.cuda.is_available():
        pg._register_backend(
            torch.device("cuda"), ProcessGroup.BackendType.GLOO, backend_class
        )
    return pg


def _nccl_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device,
    **kwargs: object,
) -> ProcessGroup:
    from torch.distributed import ProcessGroupNCCL

    opts = ProcessGroupNCCL.Options()
    opts._timeout = timeout
    for k, v in kwargs.items():
        if not hasattr(opts, k):
            raise KeyError(f"Unknown option {k}")
        setattr(opts, k, v)

    backend_class = ProcessGroupNCCL(store, rank, world_size, opts)
    backend_class._set_sequence_number_for_group()
    backend_class.eager_connect_single_device(device)

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.NCCL)
    pg._register_backend(device, ProcessGroup.BackendType.NCCL, backend_class)

    return pg


register_backend("gloo", _gloo_factory)
register_backend("nccl", _nccl_factory)


def new_group(
    backend: str,
    timeout: timedelta,
    device: Union[str, torch.device],
    **kwargs: object,
) -> ProcessGroup:
    """
    Create a new process group with the given backend and options. This group is
    independent and will not be globally registered and thus not usable via the
    standard torch.distributed.* APIs.

    Args:
        backend: The backend to use for the process group.
        timeout: The timeout for collective operations.
        device: The device to use for the process group.
        **kwargs: All remaining arguments are passed to the backend constructor.
                  See the backend specific documentation for details.

    Returns:
        A new process group.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Backend {backend} not registered")

    device = torch.device(device)

    store, rank, world_size = next(iter(rendezvous("env://")))
    store.set_timeout(timeout)

    return _BACKENDS[backend](store, rank, world_size, timeout, device, **kwargs)


def current_process_group() -> ProcessGroup:
    """
    Get the current process group. Thread local method.

    Returns:
        The current process group.
    """
    return _current_process_group()


@contextmanager
def process_group(pg: ProcessGroup) -> Generator[None, None, None]:
    """
    Context manager for process groups. Thread local method.

    Args:
        pg: The process group to use.
    """
    prev_pg = current_process_group()

    _set_process_group(pg)
    try:
        yield
    finally:
        _set_process_group(prev_pg)

```



## High-Level Overview

"""This is an experimental new API for PyTorch Distributed. This is actively in development and subject to change or deletion entirely.This is intended as a proving ground for more flexible and object oriented distributed APIs.

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ProcessGroupFactory`

**Functions defined**: `__call__`, `register_backend`, `_gloo_factory`, `_nccl_factory`, `new_group`, `current_process_group`, `process_group`

**Key imports**: Generator, contextmanager, timedelta, Protocol, Union, torch, rendezvous, ProcessGroupGloo, ProcessGroupNCCL


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Generator
- `contextlib`: contextmanager
- `datetime`: timedelta
- `typing`: Protocol, Union
- `torch`
- `torch.distributed.rendezvous`: rendezvous
- `torch.distributed`: ProcessGroupGloo


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

Files in the same folder (`torch/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_mesh_layout.py_docs.md`](./_mesh_layout.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`c10d_logger.py_docs.md`](./c10d_logger.py_docs.md)
- [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`CONTRIBUTING.md_docs.md`](./CONTRIBUTING.md_docs.md)
- [`_functional_collectives_impl.py_docs.md`](./_functional_collectives_impl.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)


## Cross-References

- **File Documentation**: `_dist2.py_docs.md`
- **Keyword Index**: `_dist2.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
