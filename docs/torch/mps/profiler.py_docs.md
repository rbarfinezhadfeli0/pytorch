# Documentation: `torch/mps/profiler.py`

## File Metadata

- **Path**: `torch/mps/profiler.py`
- **Size**: 3,553 bytes (3.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import contextlib
from collections.abc import Iterator
from typing import Literal

import torch


__all__ = [
    "start",
    "stop",
    "profile",
    "metal_capture",
    "is_metal_capture_enabled",
    "is_capturing_metal",
]


ProfilerMode = Literal["interval", "event", "interval,event"]


def start(mode: ProfilerMode = "interval", wait_until_completed: bool = False) -> None:
    r"""Start OS Signpost tracing from MPS backend.

    The generated OS Signposts could be recorded and viewed in
    XCode Instruments Logging tool.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    mode_normalized = mode.lower().replace(" ", "")
    torch._C._mps_profilerStartTrace(  # type: ignore[attr-defined]
        mode_normalized, wait_until_completed
    )


def stop() -> None:
    r"""Stops generating OS Signpost tracing from MPS backend."""
    torch._C._mps_profilerStopTrace()  # type: ignore[attr-defined]


@contextlib.contextmanager
def profile(
    mode: ProfilerMode = "interval", wait_until_completed: bool = False
) -> Iterator[None]:
    r"""Context Manager to enabling generating OS Signpost tracing from MPS backend.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    try:
        start(mode, wait_until_completed)
        yield
    finally:
        stop()


def is_metal_capture_enabled() -> bool:
    """Checks if `metal_capture` context manager is usable
    To enable metal capture, set MTL_CAPTURE_ENABLED envvar
    """
    return torch._C._mps_isCaptureEnabled()  # type: ignore[attr-defined, no-any-return]


def is_capturing_metal() -> bool:
    """Checks if metal capture is in progress"""
    return torch._C._mps_isCapturing()  # type: ignore[attr-defined, no-any-return]


@contextlib.contextmanager
def metal_capture(fname: str) -> Iterator[None]:
    """Context manager that enables capturing of Metal calls into gputrace"""
    try:
        torch._C._mps_startCapture(fname)  # type: ignore[attr-defined]
        yield
        # Drain all the work that were enqueued during the context call
        torch.mps.synchronize()
    finally:
        torch._C._mps_stopCapture()  # type: ignore[attr-defined]

```



## High-Level Overview

r"""Start OS Signpost tracing from MPS backend.    The generated OS Signposts could be recorded and viewed in    XCode Instruments Logging tool.    Args:        mode(str): OS Signpost tracing mode could be "interval", "event",            or both "interval,event".            The interval mode traces the duration of execution of the operations,            whereas event mode marks the completion of executions.            See document `Recording Performance Data`_ for more info.        wait_until_completed(bool): Waits until the MPS Stream complete            executing each encoded GPU operation. This helps generating single            dispatches on the trace's timeline.            Note that enabling this option would affect the performance negatively.    .. _Recording Performance Data:       https://developer.apple.com/documentation/os/logging/recording_performance_data

This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `start`, `stop`, `profile`, `is_metal_capture_enabled`, `is_capturing_metal`, `metal_capture`

**Key imports**: contextlib, Iterator, Literal, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/mps`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `collections.abc`: Iterator
- `typing`: Literal
- `torch`


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

Files in the same folder (`torch/mps`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`event.py_docs.md`](./event.py_docs.md)


## Cross-References

- **File Documentation**: `profiler.py_docs.md`
- **Keyword Index**: `profiler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
