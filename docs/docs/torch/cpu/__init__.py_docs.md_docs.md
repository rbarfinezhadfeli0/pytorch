# Documentation: `docs/torch/cpu/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/cpu/__init__.py_docs.md`
- **Size**: 7,318 bytes (7.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/cpu/__init__.py`

## File Metadata

- **Path**: `torch/cpu/__init__.py`
- **Size**: 4,830 bytes (4.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
r"""
This package implements abstractions found in ``torch.cuda``
to facilitate writing device-agnostic code.
"""

from contextlib import AbstractContextManager
from typing import Any, Optional, Union

import torch

from .. import device as _device
from . import amp


__all__ = [
    "is_available",
    "is_initialized",
    "synchronize",
    "current_device",
    "current_stream",
    "stream",
    "set_device",
    "device_count",
    "Stream",
    "StreamContext",
    "Event",
]


def _is_avx2_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX2."""
    return torch._C._cpu._is_avx2_supported()


def _is_avx512_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512."""
    return torch._C._cpu._is_avx512_supported()


def _is_avx512_bf16_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512_BF16."""
    return torch._C._cpu._is_avx512_bf16_supported()


def _is_vnni_supported() -> bool:
    r"""Returns a bool indicating if CPU supports VNNI."""
    # Note: Currently, it only checks avx512_vnni, will add the support of avx2_vnni later.
    return torch._C._cpu._is_avx512_vnni_supported()


def _is_amx_tile_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AMX_TILE."""
    return torch._C._cpu._is_amx_tile_supported()


def _is_amx_fp16_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AMX FP16."""
    return torch._C._cpu._is_amx_fp16_supported()


def _init_amx() -> bool:
    r"""Initializes AMX instructions."""
    return torch._C._cpu._init_amx()


def is_available() -> bool:
    r"""Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return True


def synchronize(device: torch.types.Device = None) -> None:
    r"""Waits for all kernels in all streams on the CPU device to complete.

    Args:
        device (torch.device or int, optional): ignored, there's only one CPU device.

    N.B. This function only exists to facilitate device-agnostic code.
    """


class Stream:
    """
    N.B. This class only exists to facilitate device-agnostic code
    """

    def __init__(self, priority: int = -1) -> None:
        pass

    def wait_stream(self, stream) -> None:
        pass

    def record_event(self) -> None:
        pass

    def wait_event(self, event) -> None:
        pass


class Event:
    def query(self) -> bool:
        return True

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass


_default_cpu_stream = Stream()
_current_stream = _default_cpu_stream


def current_stream(device: torch.types.Device = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): Ignored.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return _current_stream


class StreamContext(AbstractContextManager):
    r"""Context-manager that selects a given stream.

    N.B. This class only exists to facilitate device-agnostic code

    """

    cur_stream: Optional[Stream]

    def __init__(self, stream):
        self.stream = stream
        self.prev_stream = _default_cpu_stream

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        self.prev_stream = _current_stream
        _current_stream = cur_stream

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        _current_stream = self.prev_stream


def stream(stream: Stream) -> AbstractContextManager:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return StreamContext(stream)


def device_count() -> int:
    r"""Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return 1


def set_device(device: torch.types.Device) -> None:
    r"""Sets the current device, in CPU we do nothing.

    N.B. This function only exists to facilitate device-agnostic code
    """


def current_device() -> str:
    r"""Returns current device for cpu. Always 'cpu'.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return "cpu"


def is_initialized() -> bool:
    r"""Returns True if the CPU is initialized. Always True.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return True

```



## High-Level Overview

r"""This package implements abstractions found in ``torch.cuda``to facilitate writing device-agnostic code.

This Python file contains 5 class(es) and 26 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Stream`, `Event`, `StreamContext`

**Functions defined**: `_is_avx2_supported`, `_is_avx512_supported`, `_is_avx512_bf16_supported`, `_is_vnni_supported`, `_is_amx_tile_supported`, `_is_amx_fp16_supported`, `_init_amx`, `is_available`, `synchronize`, `__init__`, `wait_stream`, `record_event`, `wait_event`, `query`, `record`, `synchronize`, `wait`, `current_stream`, `__init__`, `__enter__`

**Key imports**: AbstractContextManager, Any, Optional, Union, torch, device as _device, amp


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/cpu`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`: AbstractContextManager
- `typing`: Any, Optional, Union
- `torch`
- `..`: device as _device
- `.`: amp


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


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

Files in the same folder (`torch/cpu`):



## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/cpu`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/cpu`):

- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
