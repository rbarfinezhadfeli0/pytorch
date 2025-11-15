# Documentation: `torch/xpu/streams.py`

## File Metadata

- **Path**: `torch/xpu/streams.py`
- **Size**: 5,911 bytes (5.77 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import ctypes

import torch
from torch._utils import _dummy_type


if not hasattr(torch._C, "_XpuStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_XpuStreamBase"] = _dummy_type("_XpuStreamBase")
    torch._C.__dict__["_XpuEventBase"] = _dummy_type("_XpuEventBase")


class Stream(torch._C._XpuStreamBase):
    r"""Wrapper around a XPU stream.

    A XPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, which can be positive, 0, or negative.
            A lower number indicates a higher priority. By default, the priority is set to 0.
            If the value falls outside of the allowed priority range, it will automatically be
            mapped to the nearest valid priority (lowest for large positive numbers or
            highest for large negative numbers).
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.xpu.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (torch.xpu.Event): an event to wait for.
        """
        event.wait(self)

    def wait_stream(self, stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Record an event.

        Args:
            event (torch.xpu.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.sycl_queue)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.sycl_queue, self.device))

    def __repr__(self) -> str:
        return f"torch.xpu.Stream(device={self.device} sycl_queue={self.sycl_queue:#x})"


class Event(torch._C._XpuEventBase):
    r"""Wrapper around a XPU event.

    XPU events are synchronization markers that can be used to monitor the
    device's progress, and to synchronize XPU streams.

    The underlying XPU events are lazily initialized when the event is first
    recorded. After creation, only streams on the same device may record the
    event. However, streams on any device can wait on the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __new__(cls, enable_timing=False):
        return super().__new__(cls, enable_timing=enable_timing)

    def record(self, stream=None) -> None:
        r"""Record the event in a given stream.

        Uses ``torch.xpu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = torch.xpu.current_stream()
        super().record(stream)  # pyrefly: ignore [bad-argument-type]

    def wait(self, stream=None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Use ``torch.xpu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = torch.xpu.current_stream()
        super().wait(stream)

    def query(self) -> bool:
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.
        """
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.sycl_event)

    def __repr__(self) -> str:
        if self.sycl_event:
            return f"torch.xpu.Event(sycl_event={self.sycl_event:#x})"
        else:
            return "torch.xpu.Event(uninitialized)"

```



## High-Level Overview

r"""Wrapper around a XPU stream.    A XPU stream is a linear sequence of execution that belongs to a specific    device, independent from other streams. It supports with statement as a    context manager to ensure the operators within the with block are running    on the corresponding stream.    Args:        device(torch.device or int, optional): a device on which to allocate            the stream. If :attr:`device` is ``None`` (default) or a negative            integer, this will use the current device.        priority(int, optional): priority of the stream, which can be positive, 0, or negative.            A lower number indicates a higher priority. By default, the priority is set to 0.            If the value falls outside of the allowed priority range, it will automatically be            mapped to the nearest valid priority (lowest for large positive numbers or            highest for large negative numbers).

This Python file contains 2 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Stream`, `Event`

**Functions defined**: `__new__`, `wait_event`, `wait_stream`, `record_event`, `query`, `synchronize`, `_as_parameter_`, `__eq__`, `__hash__`, `__repr__`, `__new__`, `record`, `wait`, `query`, `elapsed_time`, `synchronize`, `_as_parameter_`, `__repr__`

**Key imports**: ctypes, torch, _dummy_type


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/xpu`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `ctypes`
- `torch`
- `torch._utils`: _dummy_type


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

Files in the same folder (`torch/xpu`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`memory.py_docs.md`](./memory.py_docs.md)
- [`_gpu_trace.py_docs.md`](./_gpu_trace.py_docs.md)
- [`random.py_docs.md`](./random.py_docs.md)


## Cross-References

- **File Documentation**: `streams.py_docs.md`
- **Keyword Index**: `streams.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
