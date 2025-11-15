# Documentation: `docs/torch/accelerator/memory.py_docs.md`

## File Metadata

- **Path**: `docs/torch/accelerator/memory.py_docs.md`
- **Size**: 12,960 bytes (12.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/accelerator/memory.py`

## File Metadata

- **Path**: `torch/accelerator/memory.py`
- **Size**: 10,494 bytes (10.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections import OrderedDict
from typing import Any

import torch

from ._utils import _device_t, _get_device_index


__all__ = [
    "empty_cache",
    "get_memory_info",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other application.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return
    torch._C._accelerator_emptyCache()


def memory_stats(device_index: _device_t = None, /) -> OrderedDict[str, Any]:
    r"""Return a dictionary of accelerator device memory allocator statistics for a given device index.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from device memory allocation.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of June 2025, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of June 2025, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed device memory allocation calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of device memory allocation calls.
    - ``"num_device_free"``: number of device memory free calls.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        OrderedDict[str, Any]: an ordered dictionary mapping statistic names to their values.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return OrderedDict()
    device_index = _get_device_index(device_index, optional=True)
    stats = torch._C._accelerator_getDeviceStats(device_index)
    flat_stats = []

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                nested_prefix = f"{prefix}.{k}" if prefix else k
                flatten(nested_prefix, v)
        else:
            flat_stats.append((prefix, value))

    flatten("", stats)
    flat_stats.sort()
    # pyrefly: ignore [no-matching-overload]
    return OrderedDict(flat_stats)


def memory_allocated(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory occupied by tensors
    in bytes for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory occupied by live tensors (in bytes) within the current process.
    """
    return memory_stats(device_index).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory occupied by tensors
    in bytes for a given device index.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory occupied by live tensors (in bytes) within the current process.
    """
    return memory_stats(device_index).get("allocated_bytes.all.peak", 0)


def memory_reserved(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory managed by the caching allocator
    in bytes for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory reserved by PyTorch (in bytes) within the current process.
    """
    return memory_stats(device_index).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory managed by the caching allocator
    in bytes for a given device index.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory reserved by PyTorch (in bytes) within the current process.
    """
    return memory_stats(device_index).get("reserved_bytes.all.peak", 0)


def reset_accumulated_memory_stats(device_index: _device_t = None, /) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the current :ref:`accelerator<accelerators>`
    memory allocator for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    device_index = _get_device_index(device_index, optional=True)
    return torch._C._accelerator_resetAccumulatedStats(device_index)


def reset_peak_memory_stats(device_index: _device_t = None, /) -> None:
    r"""Reset the "peak" stats tracked by the current :ref:`accelerator<accelerators>`
    memory allocator for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    device_index = _get_device_index(device_index, optional=True)
    return torch._C._accelerator_resetPeakStats(device_index)


def get_memory_info(device_index: _device_t = None, /) -> tuple[int, int]:
    r"""Return the current device memory information for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        tuple[int, int]: a tuple of two integers (free_memory, total_memory) in bytes.
            The first value is the free memory on the device (available across all processes and applications),
            The second value is the device's total hardware memory capacity.
    """
    device_index = _get_device_index(device_index, optional=True)
    return torch._C._accelerator_getMemoryInfo(device_index)

```



## High-Level Overview

r"""Release all unoccupied cached memory currently held by the caching    allocator so that those can be used in other application.    .. note:: This function is a no-op if the memory allocator for the current        :ref:`accelerator <accelerators>` has not been initialized.

This Python file contains 0 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `empty_cache`, `memory_stats`, `flatten`, `memory_allocated`, `max_memory_allocated`, `memory_reserved`, `max_memory_reserved`, `reset_accumulated_memory_stats`, `reset_peak_memory_stats`, `get_memory_info`

**Key imports**: OrderedDict, Any, torch, _device_t, _get_device_index


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/accelerator`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: OrderedDict
- `typing`: Any
- `torch`
- `._utils`: _device_t, _get_device_index


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/accelerator`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)


## Cross-References

- **File Documentation**: `memory.py_docs.md`
- **Keyword Index**: `memory.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/accelerator`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/accelerator`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/accelerator`):

- [`memory.py_kw.md_docs.md`](./memory.py_kw.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `memory.py_docs.md_docs.md`
- **Keyword Index**: `memory.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
