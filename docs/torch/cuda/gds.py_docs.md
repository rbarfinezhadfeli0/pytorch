# Documentation: `torch/cuda/gds.py`

## File Metadata

- **Path**: `torch/cuda/gds.py`
- **Size**: 5,833 bytes (5.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import os
import sys
from collections.abc import Callable
from typing import Optional

import torch
from torch.types import Storage


__all__: list[str] = [
    "gds_register_buffer",
    "gds_deregister_buffer",
    "GdsFile",
]


def _dummy_fn(name: str) -> Callable:
    def fn(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"torch._C.{name} is not supported on this platform")

    return fn


if not hasattr(torch._C, "_gds_register_buffer"):
    assert not hasattr(torch._C, "_gds_deregister_buffer")
    assert not hasattr(torch._C, "_gds_register_handle")
    assert not hasattr(torch._C, "_gds_deregister_handle")
    assert not hasattr(torch._C, "_gds_load_storage")
    assert not hasattr(torch._C, "_gds_save_storage")
    # Define functions
    torch._C.__dict__["_gds_register_buffer"] = _dummy_fn("_gds_register_buffer")
    torch._C.__dict__["_gds_deregister_buffer"] = _dummy_fn("_gds_deregister_buffer")
    torch._C.__dict__["_gds_register_handle"] = _dummy_fn("_gds_register_handle")
    torch._C.__dict__["_gds_deregister_handle"] = _dummy_fn("_gds_deregister_handle")
    torch._C.__dict__["_gds_load_storage"] = _dummy_fn("_gds_load_storage")
    torch._C.__dict__["_gds_save_storage"] = _dummy_fn("_gds_save_storage")


def gds_register_buffer(s: Storage) -> None:
    """Registers a storage on a CUDA device as a cufile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_register_buffer(s)


def gds_deregister_buffer(s: Storage) -> None:
    """Deregisters a previously registered storage on a CUDA device as a cufile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)
        >>> gds_deregister_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_deregister_buffer(s)


class GdsFile:
    r"""Wrapper around cuFile.

    cuFile is a file-like interface to the GPUDirect Storage (GDS) API.

    See the `cufile docs <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api>`_
    for more details.

    Args:
        filename (str): Name of the file to open.
        flags (int): Flags to pass to ``os.open`` when opening the file. ``os.O_DIRECT`` will
            be added automatically.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src1 = torch.randn(1024, device="cuda")
        >>> src2 = torch.randn(2, 1024, device="cuda")
        >>> file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
        >>> file.save_storage(src1.untyped_storage(), offset=0)
        >>> file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
        >>> dest1 = torch.empty(1024, device="cuda")
        >>> dest2 = torch.empty(2, 1024, device="cuda")
        >>> file.load_storage(dest1.untyped_storage(), offset=0)
        >>> file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
        >>> torch.equal(src1, dest1)
        True
        >>> torch.equal(src2, dest2)
        True

    """

    def __init__(self, filename: str, flags: int):
        if sys.platform == "win32":
            raise RuntimeError("GdsFile is not supported on this platform.")
        self.filename = filename
        self.flags = flags
        self.fd = os.open(filename, flags | os.O_DIRECT)  # type: ignore[attr-defined]
        self.handle: Optional[int] = None
        self.register_handle()

    def __del__(self) -> None:
        if self.handle is not None:
            self.deregister_handle()
        os.close(self.fd)

    def register_handle(self) -> None:
        """Registers file descriptor to cuFile Driver.

        This is a wrapper around ``cuFileHandleRegister``.
        """
        assert self.handle is None, (
            "Cannot register a handle that is already registered."
        )
        self.handle = torch._C._gds_register_handle(self.fd)

    def deregister_handle(self) -> None:
        """Deregisters file descriptor from cuFile Driver.

        This is a wrapper around ``cuFileHandleDeregister``.
        """
        assert self.handle is not None, (
            "Cannot deregister a handle that is not registered."
        )
        torch._C._gds_deregister_handle(self.handle)
        self.handle = None

    def load_storage(self, storage: Storage, offset: int = 0) -> None:
        """Loads data from the file into the storage.

        This is a wrapper around ``cuFileRead``. ``storage.nbytes()`` of data
        will be loaded from the file at ``offset`` into the storage.

        Args:
            storage (Storage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from. (Default: 0)
        """
        assert self.handle is not None, (
            "Cannot load data from a file that is not registered."
        )
        torch._C._gds_load_storage(self.handle, storage, offset)

    def save_storage(self, storage: Storage, offset: int = 0) -> None:
        """Saves data from the storage into the file.

        This is a wrapper around ``cuFileWrite``. All bytes of the storage
        will be written to the file at ``offset``.

        Args:
            storage (Storage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to. (Default: 0)
        """
        assert self.handle is not None, (
            "Cannot save data to a file that is not registered."
        )
        torch._C._gds_save_storage(self.handle, storage, offset)

```



## High-Level Overview

"""Registers a storage on a CUDA device as a cufile buffer.    Example::        >>> # xdoctest: +SKIP("gds filesystem requirements")        >>> src = torch.randn(1024, device="cuda")        >>> s = src.untyped_storage()        >>> gds_register_buffer(s)    Args:        s (Storage): Buffer to register.

This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GdsFile`

**Functions defined**: `_dummy_fn`, `fn`, `gds_register_buffer`, `gds_deregister_buffer`, `__init__`, `__del__`, `register_handle`, `deregister_handle`, `load_storage`, `save_storage`

**Key imports**: os, sys, Callable, Optional, torch, Storage


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `collections.abc`: Callable
- `typing`: Optional
- `torch`
- `torch.types`: Storage


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/cuda`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`nccl.py_docs.md`](./nccl.py_docs.md)
- [`streams.py_docs.md`](./streams.py_docs.md)
- [`jiterator.py_docs.md`](./jiterator.py_docs.md)
- [`_sanitizer.py_docs.md`](./_sanitizer.py_docs.md)
- [`graphs.py_docs.md`](./graphs.py_docs.md)
- [`_pin_memory_utils.py_docs.md`](./_pin_memory_utils.py_docs.md)
- [`_device_limits.py_docs.md`](./_device_limits.py_docs.md)
- [`green_contexts.py_docs.md`](./green_contexts.py_docs.md)


## Cross-References

- **File Documentation**: `gds.py_docs.md`
- **Keyword Index**: `gds.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
