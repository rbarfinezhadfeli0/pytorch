# Documentation: `docs/torch/_C/_autograd.pyi_docs.md`

## File Metadata

- **Path**: `docs/torch/_C/_autograd.pyi_docs.md`
- **Size**: 7,257 bytes (7.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_C/_autograd.pyi`

## File Metadata

- **Path**: `torch/_C/_autograd.pyi`
- **Size**: 4,906 bytes (4.79 KB)
- **Type**: Python Type Stub
- **Extension**: `.pyi`

## File Purpose

This is a python type stub that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from torch._C._profiler import (
    _ProfilerEvent,
    ActiveProfilerType,
    ProfilerActivity,
    ProfilerConfig,
)

# Defined in torch/csrc/autograd/init.cpp

class DeviceType(Enum):
    CPU = ...
    CUDA = ...
    XPU = ...
    MKLDNN = ...
    OPENGL = ...
    OPENCL = ...
    IDEEP = ...
    HIP = ...
    FPGA = ...
    MAIA = ...
    XLA = ...
    MTIA = ...
    MPS = ...
    HPU = ...
    Meta = ...
    Vulkan = ...
    Metal = ...
    PrivateUse1 = ...

class ProfilerEvent:
    def cpu_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def cpu_memory_usage(self) -> int: ...
    def cuda_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def privateuse1_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def cuda_memory_usage(self) -> int: ...
    def device(self) -> int: ...
    def handle(self) -> int: ...
    def has_cuda(self) -> bool: ...
    def is_remote(self) -> bool: ...
    def kind(self) -> int: ...
    def name(self) -> str: ...
    def node_id(self) -> int: ...
    def sequence_nr(self) -> int: ...
    def shapes(self) -> list[list[int]]: ...
    def thread_id(self) -> int: ...
    def flops(self) -> float: ...
    def is_async(self) -> bool: ...

class _KinetoEvent:
    def name(self) -> str: ...
    def overload_name(self) -> str: ...
    def device_index(self) -> int: ...
    def device_resource_id(self) -> int: ...
    def start_ns(self) -> int: ...
    def end_ns(self) -> int: ...
    def duration_ns(self) -> int: ...
    def is_async(self) -> bool: ...
    def linked_correlation_id(self) -> int: ...
    def shapes(self) -> list[list[int]]: ...
    def dtypes(self) -> list[str]: ...
    def concrete_inputs(self) -> list[Any]: ...
    def kwinputs(self) -> dict[str, Any]: ...
    def device_type(self) -> DeviceType: ...
    def start_thread_id(self) -> int: ...
    def end_thread_id(self) -> int: ...
    def correlation_id(self) -> int: ...
    def fwd_thread_id(self) -> int: ...
    def stack(self) -> list[str]: ...
    def scope(self) -> int: ...
    def sequence_nr(self) -> int: ...
    def flops(self) -> int: ...
    def cuda_elapsed_us(self) -> int: ...
    def privateuse1_elapsed_us(self) -> int: ...
    def is_user_annotation(self) -> bool: ...
    def is_hidden_event(self) -> bool: ...
    def metadata_json(self) -> str: ...

class _ProfilerResult:
    def events(self) -> list[_KinetoEvent]: ...
    def legacy_events(self) -> list[list[ProfilerEvent]]: ...
    def save(self, path: str) -> None: ...
    def experimental_event_tree(self) -> list[_ProfilerEvent]: ...
    def trace_start_ns(self) -> int: ...

class SavedTensor: ...

def _enable_profiler(
    config: ProfilerConfig,
    activities: set[ProfilerActivity],
) -> None: ...
def _prepare_profiler(
    config: ProfilerConfig,
    activities: set[ProfilerActivity],
) -> None: ...
def _toggle_collection_dynamic(
    enable: bool,
    activities: set[ProfilerActivity],
) -> None: ...
def _disable_profiler() -> _ProfilerResult: ...
def _profiler_enabled() -> bool: ...
def _add_metadata_json(key: str, value: str) -> None: ...
def _kineto_step() -> None: ...
def _get_current_graph_task_keep_graph() -> bool: ...
def _get_sequence_nr() -> int: ...
def kineto_available() -> bool: ...
def _record_function_with_args_enter(name: str, *args) -> torch.Tensor: ...
def _record_function_with_args_exit(handle: torch.Tensor) -> None: ...
def _supported_activities() -> set[ProfilerActivity]: ...
def _enable_record_function(enable: bool) -> None: ...
def _set_empty_test_observer(is_global: bool, sampling_prob: float) -> None: ...
def _push_saved_tensors_default_hooks(
    pack_hook: Callable[[torch.Tensor], Any],
    unpack_hook: Callable[[Any], torch.Tensor],
) -> None: ...
def _pop_saved_tensors_default_hooks() -> None: ...
def _top_saved_tensors_default_hooks(
    ignore_is_tracing: bool,
) -> tuple[Callable[[torch.Tensor], Any], Callable[[Any], torch.Tensor]]: ...
def _unsafe_set_version_counter(
    t: tuple[torch.Tensor, ...], prev_version: tuple[int, ...]
) -> None: ...
def _enable_profiler_legacy(config: ProfilerConfig) -> None: ...
def _disable_profiler_legacy() -> list[list[ProfilerEvent]]: ...
def _profiler_type() -> ActiveProfilerType: ...
def _saved_tensors_hooks_enable() -> None: ...
def _saved_tensors_hooks_disable(message: str, fail_if_non_empty=True) -> None: ...
def _saved_tensors_hooks_get_disabled_error_message() -> str | None: ...
def _saved_tensors_hooks_set_tracing(is_tracing: bool) -> bool: ...

class CreationMeta(Enum):
    DEFAULT = ...
    IN_CUSTOM_FUNCTION = ...
    MULTI_OUTPUT_NODE = ...
    NO_GRAD_MODE = ...
    INFERENCE_MODE = ...

def _set_creation_meta(t: torch.Tensor, creation_meta: CreationMeta) -> None: ...
def _get_creation_meta(t: torch.Tensor) -> CreationMeta: ...

```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/_C`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_C`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`torch/_C`):

- [`_onnx.pyi_docs.md`](./_onnx.pyi_docs.md)
- [`_distributed_autograd.pyi_docs.md`](./_distributed_autograd.pyi_docs.md)
- [`_distributed_rpc_testing.pyi_docs.md`](./_distributed_rpc_testing.pyi_docs.md)
- [`_distributed_rpc.pyi_docs.md`](./_distributed_rpc.pyi_docs.md)
- [`_lazy_ts_backend.pyi_docs.md`](./_lazy_ts_backend.pyi_docs.md)
- [`_verbose.pyi_docs.md`](./_verbose.pyi_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`_instruction_counter.pyi_docs.md`](./_instruction_counter.pyi_docs.md)
- [`_functions.pyi_docs.md`](./_functions.pyi_docs.md)
- [`_aoti.pyi_docs.md`](./_aoti.pyi_docs.md)


## Cross-References

- **File Documentation**: `_autograd.pyi_docs.md`
- **Keyword Index**: `_autograd.pyi_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_C`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_C`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/_C`):

- [`_nvtx.pyi_docs.md_docs.md`](./_nvtx.pyi_docs.md_docs.md)
- [`_aoti.pyi_docs.md_docs.md`](./_aoti.pyi_docs.md_docs.md)
- [`_cpu.pyi_docs.md_docs.md`](./_cpu.pyi_docs.md_docs.md)
- [`_lazy_ts_backend.pyi_docs.md_docs.md`](./_lazy_ts_backend.pyi_docs.md_docs.md)
- [`_distributed_c10d.pyi_kw.md_docs.md`](./_distributed_c10d.pyi_kw.md_docs.md)
- [`_profiler.pyi_docs.md_docs.md`](./_profiler.pyi_docs.md_docs.md)
- [`_functionalization.pyi_kw.md_docs.md`](./_functionalization.pyi_kw.md_docs.md)
- [`_distributed.pyi_docs.md_docs.md`](./_distributed.pyi_docs.md_docs.md)
- [`_itt.pyi_docs.md_docs.md`](./_itt.pyi_docs.md_docs.md)
- [`build.bzl_kw.md_docs.md`](./build.bzl_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_autograd.pyi_docs.md_docs.md`
- **Keyword Index**: `_autograd.pyi_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
