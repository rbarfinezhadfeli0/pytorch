# Documentation: `docs/torch/_C/_aoti.pyi_docs.md`

## File Metadata

- **Path**: `docs/torch/_C/_aoti.pyi_docs.md`
- **Size**: 8,318 bytes (8.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_C/_aoti.pyi`

## File Metadata

- **Path**: `torch/_C/_aoti.pyi`
- **Size**: 6,015 bytes (5.87 KB)
- **Type**: Python Type Stub
- **Extension**: `.pyi`

## File Purpose

This is a python type stub that is part of the PyTorch project.

## Original Source

```python
from ctypes import c_void_p
from typing import overload, Protocol

from torch import Tensor

# Defined in torch/csrc/inductor/aoti_runner/pybind.cpp

# Tensor to AtenTensorHandle
def unsafe_alloc_void_ptrs_from_tensors(tensors: list[Tensor]) -> list[c_void_p]: ...
def unsafe_alloc_void_ptr_from_tensor(tensor: Tensor) -> c_void_p: ...

# AtenTensorHandle to Tensor
def alloc_tensors_by_stealing_from_void_ptrs(
    handles: list[c_void_p],
) -> list[Tensor]: ...
def alloc_tensor_by_stealing_from_void_ptr(
    handle: c_void_p,
) -> Tensor: ...

class AOTIModelContainerRunner(Protocol):
    def run(
        self, inputs: list[Tensor], stream_handle: c_void_p = ...
    ) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_names_to_original_fqns(self) -> dict[str, str]: ...
    def get_constant_names_to_dtypes(self) -> dict[str, int]: ...
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]: ...
    def update_constant_buffer(
        self,
        tensor_map: dict[str, Tensor],
        use_inactive: bool,
        validate_full_updates: bool,
        user_managed: bool = ...,
    ) -> None: ...
    def swap_constant_buffer(self) -> None: ...
    def free_inactive_constant_buffer(self) -> None: ...

class AOTIModelContainerRunnerCpu:
    def __init__(self, model_so_path: str, num_models: int) -> None: ...
    def run(
        self, inputs: list[Tensor], stream_handle: c_void_p = ...
    ) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_names_to_original_fqns(self) -> dict[str, str]: ...
    def get_constant_names_to_dtypes(self) -> dict[str, int]: ...
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]: ...
    def update_constant_buffer(
        self,
        tensor_map: dict[str, Tensor],
        use_inactive: bool,
        validate_full_updates: bool,
        user_managed: bool = ...,
    ) -> None: ...
    def swap_constant_buffer(self) -> None: ...
    def free_inactive_constant_buffer(self) -> None: ...

class AOTIModelContainerRunnerCuda:
    @overload
    def __init__(self, model_so_path: str, num_models: int) -> None: ...
    @overload
    def __init__(
        self, model_so_path: str, num_models: int, device_str: str
    ) -> None: ...
    @overload
    def __init__(
        self, model_so_path: str, num_models: int, device_str: str, cubin_dir: str
    ) -> None: ...
    def run(
        self, inputs: list[Tensor], stream_handle: c_void_p = ...
    ) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_names_to_original_fqns(self) -> dict[str, str]: ...
    def get_constant_names_to_dtypes(self) -> dict[str, int]: ...
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]: ...
    def update_constant_buffer(
        self,
        tensor_map: dict[str, Tensor],
        use_inactive: bool,
        validate_full_updates: bool,
        user_managed: bool = ...,
    ) -> None: ...
    def swap_constant_buffer(self) -> None: ...
    def free_inactive_constant_buffer(self) -> None: ...

class AOTIModelContainerRunnerXpu:
    @overload
    def __init__(self, model_so_path: str, num_models: int) -> None: ...
    @overload
    def __init__(
        self, model_so_path: str, num_models: int, device_str: str
    ) -> None: ...
    @overload
    def __init__(
        self, model_so_path: str, num_models: int, device_str: str, kernel_bin_dir: str
    ) -> None: ...
    def run(
        self, inputs: list[Tensor], stream_handle: c_void_p = ...
    ) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_names_to_original_fqns(self) -> dict[str, str]: ...
    def get_constant_names_to_dtypes(self) -> dict[str, int]: ...
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]: ...
    def update_constant_buffer(
        self,
        tensor_map: dict[str, Tensor],
        use_inactive: bool,
        validate_full_updates: bool,
        user_managed: bool = ...,
    ) -> None: ...
    def swap_constant_buffer(self) -> None: ...
    def free_inactive_constant_buffer(self) -> None: ...

class AOTIModelContainerRunnerMps:
    def __init__(self, model_so_path: str, num_models: int) -> None: ...
    def run(
        self, inputs: list[Tensor], stream_handle: c_void_p = ...
    ) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_names_to_original_fqns(self) -> dict[str, str]: ...
    def get_constant_names_to_dtypes(self) -> dict[str, int]: ...
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]: ...
    def update_constant_buffer(
        self,
        tensor_map: dict[str, Tensor],
        use_inactive: bool,
        validate_full_updates: bool,
        user_managed: bool = ...,
    ) -> None: ...
    def swap_constant_buffer(self) -> None: ...
    def free_inactive_constant_buffer(self) -> None: ...

# Defined in torch/csrc/inductor/aoti_package/pybind.cpp
class AOTIModelPackageLoader:
    def __init__(
        self,
        model_package_path: str,
        model_name: str,
        run_single_threaded: bool,
        num_runners: int,
        device_index: int,
    ) -> None: ...
    def get_metadata(self) -> dict[str, str]: ...
    def run(
        self, inputs: list[Tensor], stream_handle: c_void_p = ...
    ) -> list[Tensor]: ...
    def boxed_run(
        self, inputs: list[Tensor], stream_handle: c_void_p = ...
    ) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_fqns(self) -> list[str]: ...
    def load_constants(
        self,
        constants_map: dict[str, Tensor],
        use_inactive: bool,
        check_full_update: bool,
        user_managed: bool = ...,
    ) -> None: ...
    def update_constant_buffer(
        self,
        tensor_map: dict[str, Tensor],
        use_inactive: bool,
        validate_full_updates: bool,
        user_managed: bool = ...,
    ) -> None: ...

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


## Cross-References

- **File Documentation**: `_aoti.pyi_docs.md`
- **Keyword Index**: `_aoti.pyi_kw.md`
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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_C`):

- [`_nvtx.pyi_docs.md_docs.md`](./_nvtx.pyi_docs.md_docs.md)
- [`_cpu.pyi_docs.md_docs.md`](./_cpu.pyi_docs.md_docs.md)
- [`_lazy_ts_backend.pyi_docs.md_docs.md`](./_lazy_ts_backend.pyi_docs.md_docs.md)
- [`_distributed_c10d.pyi_kw.md_docs.md`](./_distributed_c10d.pyi_kw.md_docs.md)
- [`_profiler.pyi_docs.md_docs.md`](./_profiler.pyi_docs.md_docs.md)
- [`_functionalization.pyi_kw.md_docs.md`](./_functionalization.pyi_kw.md_docs.md)
- [`_distributed.pyi_docs.md_docs.md`](./_distributed.pyi_docs.md_docs.md)
- [`_itt.pyi_docs.md_docs.md`](./_itt.pyi_docs.md_docs.md)
- [`build.bzl_kw.md_docs.md`](./build.bzl_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_aoti.pyi_docs.md_docs.md`
- **Keyword Index**: `_aoti.pyi_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
