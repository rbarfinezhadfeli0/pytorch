# Documentation: `docs/torch/cuda/nvtx.py_docs.md`

## File Metadata

- **Path**: `docs/torch/cuda/nvtx.py_docs.md`
- **Size**: 6,781 bytes (6.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/cuda/nvtx.py`

## File Metadata

- **Path**: `torch/cuda/nvtx.py`
- **Size**: 3,705 bytes (3.62 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
r"""This package adds support for NVIDIA Tools Extension (NVTX) used in profiling."""

from contextlib import contextmanager


try:
    from torch._C import _nvtx
except ImportError:

    class _NVTXStub:
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError(
                "NVTX functions not installed. Are you sure you have a CUDA build?"
            )

        rangePushA = _fail
        rangePop = _fail
        markA = _fail

    _nvtx = _NVTXStub()  # type: ignore[assignment]

__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]


def range_push(msg):
    """
    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
    """
    return _nvtx.rangePushA(msg)


def range_pop():
    """Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended."""
    return _nvtx.rangePop()


def range_start(msg) -> int:
    """
    Mark the start of a range with string message. It returns an unique handle
    for this range to pass to the corresponding call to rangeEnd().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns: A range handle (uint64_t) that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
    """
    # pyrefly: ignore [missing-attribute]
    return _nvtx.rangeStartA(msg)


def range_end(range_id) -> None:
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): an unique handle for the start range.
    """
    # pyrefly: ignore [missing-attribute]
    _nvtx.rangeEnd(range_id)


def _device_range_start(msg: str, stream: int = 0) -> object:
    """
    Marks the start of a range with string message.
    It returns an opaque heap-allocated handle for this range
    to pass to the corresponding call to device_range_end().

    A key difference between this and range_start is that the
    range_start marks the range right away, while _device_range_start
    marks the start of the range as soon as all the tasks on the
    CUDA stream are completed.

    Returns: An opaque heap-allocated handle that should be passed to _device_range_end().

    Args:
        msg (str): ASCII message to associate with the range.
        stream (int): CUDA stream id.
    """
    # pyrefly: ignore [missing-attribute]
    return _nvtx.deviceRangeStart(msg, stream)


def _device_range_end(range_handle: object, stream: int = 0) -> None:
    """
    Mark the end of a range for a given range_handle as soon as all the tasks
    on the CUDA stream are completed.

    Args:
        range_handle: an unique handle for the start range.
        stream (int): CUDA stream id.
    """
    # pyrefly: ignore [missing-attribute]
    _nvtx.deviceRangeEnd(range_handle, stream)


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
    return _nvtx.markA(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()

```



## High-Level Overview

r"""This package adds support for NVIDIA Tools Extension (NVTX) used in profiling."""from contextlib import contextmanagertry:    from torch._C import _nvtxexcept ImportError:    class _NVTXStub:        @staticmethod        def _fail(*args, **kwargs):            raise RuntimeError(                "NVTX functions not installed. Are you sure you have a CUDA build?"            )        rangePushA = _fail        rangePop = _fail        markA = _fail    _nvtx = _NVTXStub()  # type: ignore[assignment]__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]def range_push(msg):

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_NVTXStub`

**Functions defined**: `_fail`, `range_push`, `range_pop`, `range_start`, `range_end`, `_device_range_start`, `_device_range_end`, `mark`, `range`

**Key imports**: contextmanager, _nvtx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`: contextmanager
- `torch._C`: _nvtx


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
- [`gds.py_docs.md`](./gds.py_docs.md)
- [`_pin_memory_utils.py_docs.md`](./_pin_memory_utils.py_docs.md)
- [`_device_limits.py_docs.md`](./_device_limits.py_docs.md)
- [`green_contexts.py_docs.md`](./green_contexts.py_docs.md)


## Cross-References

- **File Documentation**: `nvtx.py_docs.md`
- **Keyword Index**: `nvtx.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/cuda`):

- [`profiler.py_docs.md_docs.md`](./profiler.py_docs.md_docs.md)
- [`sparse.py_docs.md_docs.md`](./sparse.py_docs.md_docs.md)
- [`tunable.py_kw.md_docs.md`](./tunable.py_kw.md_docs.md)
- [`_pin_memory_utils.py_kw.md_docs.md`](./_pin_memory_utils.py_kw.md_docs.md)
- [`nccl.py_kw.md_docs.md`](./nccl.py_kw.md_docs.md)
- [`gds.py_kw.md_docs.md`](./gds.py_kw.md_docs.md)
- [`jiterator.py_docs.md_docs.md`](./jiterator.py_docs.md_docs.md)
- [`memory.py_kw.md_docs.md`](./memory.py_kw.md_docs.md)
- [`random.py_docs.md_docs.md`](./random.py_docs.md_docs.md)
- [`nvtx.py_kw.md_docs.md`](./nvtx.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `nvtx.py_docs.md_docs.md`
- **Keyword Index**: `nvtx.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
