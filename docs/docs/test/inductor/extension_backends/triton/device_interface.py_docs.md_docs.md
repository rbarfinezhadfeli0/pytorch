# Documentation: `docs/test/inductor/extension_backends/triton/device_interface.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/extension_backends/triton/device_interface.py_docs.md`
- **Size**: 5,599 bytes (5.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/inductor/extension_backends/triton/device_interface.py`

## File Metadata

- **Path**: `test/inductor/extension_backends/triton/device_interface.py`
- **Size**: 3,042 bytes (2.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
from __future__ import annotations

import time

import torch
from torch._dynamo import device_interface  # noqa: PLC2701 import-private-name


class DeviceProperties:
    def __init__(self) -> None:
        self.major = 8  # TODO: bypass check for H100 in triton_heuristics.py
        self.max_threads_per_multi_processor = 1
        self.multi_processor_count = 80


class DeviceInterface(device_interface.DeviceInterface):
    class Event(torch.Event):
        def __init__(
            self,
            enable_timing: bool = False,
            blocking: bool = False,
            interprocess: bool = False,
        ) -> None:
            self.enable_timing = enable_timing
            self.recorded_time: int | None = None

        def record(self, stream) -> None:
            if not self.enable_timing:
                return
            assert self.recorded_time is None
            self.recorded_time = time.perf_counter_ns()

        def elapsed_time(self, end_event: DeviceInterface.Event) -> float:
            assert self.recorded_time
            assert end_event.recorded_time
            # convert to ms
            return (end_event.recorded_time - self.recorded_time) / 1000000

        def wait(self, stream) -> None:
            pass

        def query(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

    class device:  # noqa: N801 invalid-class-name # pyright: ignore [reportIncompatibleVariableOverride]
        def __init__(self, device) -> None:
            self.device = device

    class Worker(device_interface.DeviceInterface.Worker):
        @staticmethod
        def set_device(device: int) -> None:
            # No device index for our backend
            pass

        @staticmethod
        def current_device() -> int:
            # No device index for our backend
            return 0

        @staticmethod
        def get_device_properties(
            device=None,
        ) -> DeviceProperties:
            return DeviceProperties()

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def set_device(device) -> None:
        pass

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        assert device == 0, (
            f"Only device index 0 is supported, tried to set index to {device}"
        )
        return 0  # previous device is always 0

    @staticmethod
    def exchange_device(device: int) -> int:
        assert device == 0, (
            f"Only device index 0 is supported, tried to set index to {device}"
        )
        return 0  # previous device is always 0

    @staticmethod
    def get_raw_stream(device_index: int):
        return None

    @staticmethod
    def synchronize(device) -> None:
        pass

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_compute_capability(device) -> int:
        return 0

```



## High-Level Overview


This Python file contains 5 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DeviceProperties`, `DeviceInterface`, `Event`, `device`, `Worker`

**Functions defined**: `__init__`, `__init__`, `record`, `elapsed_time`, `wait`, `query`, `synchronize`, `__init__`, `set_device`, `current_device`, `get_device_properties`, `current_device`, `set_device`, `device_count`, `maybe_exchange_device`, `exchange_device`, `get_raw_stream`, `synchronize`, `is_available`, `get_compute_capability`

**Key imports**: annotations, time, torch, device_interface  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor/extension_backends/triton`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `time`
- `torch`
- `torch._dynamo`: device_interface  


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

This is a test file. Run it with:

```bash
python test/inductor/extension_backends/triton/device_interface.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor/extension_backends/triton`):

- [`extension_codegen_backend.py_docs.md`](./extension_codegen_backend.py_docs.md)


## Cross-References

- **File Documentation**: `device_interface.py_docs.md`
- **Keyword Index**: `device_interface.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor/extension_backends/triton`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor/extension_backends/triton`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

This is a test file. Run it with:

```bash
python docs/test/inductor/extension_backends/triton/device_interface.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor/extension_backends/triton`):

- [`extension_codegen_backend.py_kw.md_docs.md`](./extension_codegen_backend.py_kw.md_docs.md)
- [`device_interface.py_kw.md_docs.md`](./device_interface.py_kw.md_docs.md)
- [`extension_codegen_backend.py_docs.md_docs.md`](./extension_codegen_backend.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `device_interface.py_docs.md_docs.md`
- **Keyword Index**: `device_interface.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
