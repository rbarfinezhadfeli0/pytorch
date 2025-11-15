# Documentation: `docs/torch/_C/_distributed_rpc.pyi_docs.md`

## File Metadata

- **Path**: `docs/torch/_C/_distributed_rpc.pyi_docs.md`
- **Size**: 8,473 bytes (8.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_C/_distributed_rpc.pyi`

## File Metadata

- **Path**: `torch/_C/_distributed_rpc.pyi`
- **Size**: 6,080 bytes (5.94 KB)
- **Type**: Python Type Stub
- **Extension**: `.pyi`

## File Purpose

This is a python type stub that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# mypy: disable-error-code="type-arg"
from datetime import timedelta
from typing import Any, Generic, overload, TypeVar

import torch
from torch._C import Future
from torch._C._autograd import ProfilerEvent
from torch._C._distributed_c10d import Store
from torch._C._profiler import ProfilerConfig

# This module is defined in torch/csrc/distributed/rpc/init.cpp

_DEFAULT_INIT_METHOD: str
_DEFAULT_NUM_WORKER_THREADS: int
_UNSET_RPC_TIMEOUT: float
_DEFAULT_RPC_TIMEOUT_SEC: float

_T = TypeVar("_T")

class RpcBackendOptions:
    rpc_timeout: float
    init_method: str
    def __init__(
        self,
        rpc_timeout: float = ...,
        init_method: str = ...,
    ) -> None: ...

class WorkerInfo:
    def __init__(self, name: str, worker_id: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def id(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class RpcAgent:
    def join(self, shutdown: bool = False, timeout: float = 0): ...
    def sync(self): ...
    def shutdown(self): ...
    @overload
    def get_worker_info(self) -> WorkerInfo: ...
    @overload
    def get_worker_info(self, workerName: str) -> WorkerInfo: ...
    def get_worker_infos(self) -> list[WorkerInfo]: ...
    def _get_device_map(self, dst: WorkerInfo) -> dict[torch.device, torch.device]: ...
    def get_debug_info(self) -> dict[str, str]: ...
    def get_metrics(self) -> dict[str, str]: ...

class PyRRef(Generic[_T]):
    def __init__(self, value: _T, type_hint: Any = None) -> None: ...
    def is_owner(self) -> bool: ...
    def confirmed_by_owner(self) -> bool: ...
    def owner(self) -> WorkerInfo: ...
    def owner_name(self) -> str: ...
    def to_here(self, timeout: float = ...) -> _T: ...
    def local_value(self) -> Any: ...
    def rpc_sync(self, timeout: float = ...) -> Any: ...
    def rpc_async(self, timeout: float = ...) -> Any: ...
    def remote(self, timeout: float = ...) -> Any: ...
    def _serialize(self) -> tuple: ...
    @staticmethod
    def _deserialize(tp: tuple) -> PyRRef: ...
    def _get_type(self) -> type[_T]: ...
    def _get_future(self) -> Future[_T]: ...
    def _get_profiling_future(self) -> Future[_T]: ...
    def _set_profiling_future(self, profilingFuture: Future[_T]): ...

class _TensorPipeRpcBackendOptionsBase(RpcBackendOptions):
    num_worker_threads: int
    device_maps: dict[str, dict[torch.device, torch.device]]
    devices: list[torch.device]
    def __init__(
        self,
        num_worker_threads: int,
        _transports: list | None,
        _channels: list | None,
        rpc_timeout: float = ...,
        init_method: str = ...,
        device_maps: dict[str, dict[torch.device, torch.device]] = {},  # noqa: B006
        devices: list[torch.device] = [],  # noqa: B006
    ) -> None: ...
    def _set_device_map(
        self,
        to: str,
        device_map: dict[torch.device, torch.device],
    ): ...

class TensorPipeAgent(RpcAgent):
    def __init__(
        self,
        store: Store,
        name: str,
        worker_id: int,
        world_size: int | None,
        opts: _TensorPipeRpcBackendOptionsBase,
        reverse_device_maps: dict[str, dict[torch.device, torch.device]],
        devices: list[torch.device],
    ) -> None: ...
    def join(self, shutdown: bool = False, timeout: float = 0): ...
    def shutdown(self): ...
    @overload
    def get_worker_info(self) -> WorkerInfo: ...
    @overload
    def get_worker_info(self, workerName: str) -> WorkerInfo: ...
    @overload
    def get_worker_info(self, id: int) -> WorkerInfo: ...
    def get_worker_infos(self) -> list[WorkerInfo]: ...
    def _get_device_map(self, dst: WorkerInfo) -> dict[torch.device, torch.device]: ...
    def _update_group_membership(
        self,
        worker_info: WorkerInfo,
        my_devices: list[torch.device],
        reverse_device_map: dict[str, dict[torch.device, torch.device]],
        is_join: bool,
    ): ...
    def _get_backend_options(self) -> _TensorPipeRpcBackendOptionsBase: ...
    @property
    def is_static_group(self) -> bool: ...
    @property
    def store(self) -> Store: ...

def _is_current_rpc_agent_set() -> bool: ...
def _get_current_rpc_agent() -> RpcAgent: ...
def _set_and_start_rpc_agent(agent: RpcAgent): ...
def _reset_current_rpc_agent(): ...
def _delete_all_user_and_unforked_owner_rrefs(timeout: timedelta = ...): ...
def _destroy_rref_context(ignoreRRefLeak: bool): ...
def _rref_context_get_debug_info() -> dict[str, str]: ...
def _cleanup_python_rpc_handler(): ...
def _invoke_rpc_builtin(
    dst: WorkerInfo,
    opName: str,
    rpcTimeoutSeconds: float,
    *args: Any,
    **kwargs: Any,
): ...
def _invoke_rpc_python_udf(
    dst: WorkerInfo,
    pickledPythonUDF: str,
    tensors: list[torch.Tensor],
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
): ...
def _invoke_rpc_torchscript(
    dstWorkerName: str,
    qualifiedNameStr: str,
    argsTuple: tuple,
    kwargsDict: dict,
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
): ...
def _invoke_remote_builtin(
    dst: WorkerInfo,
    opName: str,
    rpcTimeoutSeconds: float,
    *args: Any,
    **kwargs: Any,
): ...
def _invoke_remote_python_udf(
    dst: WorkerInfo,
    pickledPythonUDF: str,
    tensors: list[torch.Tensor],
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
): ...
def _invoke_remote_torchscript(
    dstWorkerName: WorkerInfo,
    qualifiedNameStr: str,
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
    *args: Any,
    **kwargs: Any,
): ...
def get_rpc_timeout() -> float: ...
def enable_gil_profiling(flag: bool): ...
def _set_rpc_timeout(rpcTimeoutSeconds: float): ...

class RemoteProfilerManager:
    @staticmethod
    def set_current_profiling_key(key: str): ...

def _enable_server_process_global_profiler(new_config: ProfilerConfig): ...
def _disable_server_process_global_profiler() -> list[list[list[ProfilerEvent]]]: ...
def _set_profiler_node_id(default_node_id: int): ...
def _enable_jit_rref_pickle(): ...
def _disable_jit_rref_pickle(): ...

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
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

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
- [`_lazy_ts_backend.pyi_docs.md`](./_lazy_ts_backend.pyi_docs.md)
- [`_verbose.pyi_docs.md`](./_verbose.pyi_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`_instruction_counter.pyi_docs.md`](./_instruction_counter.pyi_docs.md)
- [`_functions.pyi_docs.md`](./_functions.pyi_docs.md)
- [`_aoti.pyi_docs.md`](./_aoti.pyi_docs.md)


## Cross-References

- **File Documentation**: `_distributed_rpc.pyi_docs.md`
- **Keyword Index**: `_distributed_rpc.pyi_kw.md`
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
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

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

- **File Documentation**: `_distributed_rpc.pyi_docs.md_docs.md`
- **Keyword Index**: `_distributed_rpc.pyi_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
