# Documentation: `docs/torch/_C/_functorch.pyi_docs.md`

## File Metadata

- **Path**: `docs/torch/_C/_functorch.pyi_docs.md`
- **Size**: 5,724 bytes (5.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_C/_functorch.pyi`

## File Metadata

- **Path**: `torch/_C/_functorch.pyi`
- **Size**: 3,428 bytes (3.35 KB)
- **Type**: Python Type Stub
- **Extension**: `.pyi`

## File Purpose

This is a python type stub that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from enum import Enum

from torch import Tensor

# Defined in torch/csrc/functorch/init.cpp

def set_inplace_requires_grad_allowed(allowed: bool) -> None: ...
def get_inplace_requires_grad_allowed() -> bool: ...
def _set_dynamic_layer_keys_included(included: bool) -> None: ...
def get_unwrapped(tensor: Tensor) -> Tensor: ...
def is_batchedtensor(tensor: Tensor) -> bool: ...
def is_functionaltensor(tensor: Tensor) -> bool: ...
def is_functorch_wrapped_tensor(tensor: Tensor) -> bool: ...
def is_gradtrackingtensor(tensor: Tensor) -> bool: ...
def is_legacy_batchedtensor(tensor: Tensor) -> bool: ...
def maybe_get_bdim(tensor: Tensor) -> int: ...
def maybe_get_level(tensor: Tensor) -> int: ...
def maybe_current_level() -> int | None: ...
def unwrap_if_dead(tensor: Tensor) -> Tensor: ...
def _unwrap_for_grad(tensor: Tensor, level: int) -> Tensor: ...
def _wrap_for_grad(tensor: Tensor, level: int) -> Tensor: ...
def _unwrap_batched(tensor: Tensor, level: int) -> tuple[Tensor, int | None]: ...
def current_level() -> int: ...
def count_jvp_interpreters() -> int: ...
def _add_batch_dim(tensor: Tensor, bdim: int, level: int) -> Tensor: ...
def _maybe_unsafe_set_level(tensor: Tensor, level: int) -> None: ...
def set_single_level_autograd_function_allowed(allowed: bool) -> None: ...
def get_single_level_autograd_function_allowed() -> bool: ...
def _unwrap_functional_tensor(tensor: Tensor, reapply_views: bool) -> Tensor: ...
def _wrap_functional_tensor(tensor: Tensor, level: int) -> Tensor: ...
def _vmap_increment_nesting(batch_size: int, randomness: str) -> int: ...
def _vmap_decrement_nesting() -> int: ...
def _grad_increment_nesting() -> int: ...
def _grad_decrement_nesting() -> int: ...
def _jvp_increment_nesting() -> int: ...
def _jvp_decrement_nesting() -> int: ...

# Defined in aten/src/ATen/functorch/Interpreter.h
class TransformType(Enum):
    Torch = ...
    Vmap = ...
    Grad = ...
    Jvp = ...
    Functionalize = ...

class RandomnessType(Enum):
    Error = ...
    Same = ...
    Different = ...

class CInterpreter:
    def key(self) -> TransformType: ...
    def level(self) -> int: ...
    def serialize(self) -> bytes: ...
    @staticmethod
    def deserialize(bytes) -> CInterpreter: ...

class CGradInterpreterPtr:
    def __init__(self, interpreter: CInterpreter) -> None: ...
    def lift(self, Tensor) -> Tensor: ...
    def prevGradMode(self) -> bool: ...

class CJvpInterpreterPtr:
    def __init__(self, interpreter: CInterpreter) -> None: ...
    def lift(self, Tensor) -> Tensor: ...
    def prevFwdGradMode(self) -> bool: ...

class CFunctionalizeInterpreterPtr:
    def __init__(self, interpreter: CInterpreter) -> None: ...
    def key(self) -> TransformType: ...
    def level(self) -> int: ...
    def functionalizeAddBackViews(self) -> bool: ...

class CVmapInterpreterPtr:
    def __init__(self, interpreter: CInterpreter) -> None: ...
    def key(self) -> TransformType: ...
    def level(self) -> int: ...
    def batchSize(self) -> int: ...
    def randomness(self) -> RandomnessType: ...

class DynamicLayer: ...

def get_dynamic_layer_stack_depth() -> int: ...
def get_interpreter_stack() -> list[CInterpreter]: ...
def peek_interpreter_stack() -> CInterpreter: ...
def pop_dynamic_layer_stack() -> DynamicLayer: ...
def pop_dynamic_layer_stack_and_undo_to_depth(int) -> None: ...
def push_dynamic_layer_stack(dl: DynamicLayer) -> int: ...

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

- **File Documentation**: `_functorch.pyi_docs.md`
- **Keyword Index**: `_functorch.pyi_kw.md`
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

- **File Documentation**: `_functorch.pyi_docs.md_docs.md`
- **Keyword Index**: `_functorch.pyi_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
