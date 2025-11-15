# Documentation: `docs/torch/distributed/optim/zero_redundancy_optimizer.pyi_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/optim/zero_redundancy_optimizer.pyi_docs.md`
- **Size**: 5,224 bytes (5.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/optim/zero_redundancy_optimizer.pyi`

## File Metadata

- **Path**: `torch/distributed/optim/zero_redundancy_optimizer.pyi`
- **Size**: 2,834 bytes (2.77 KB)
- **Type**: Python Type Stub
- **Extension**: `.pyi`

## File Purpose

This is a python type stub that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import enum
from collections.abc import Callable
from typing import Any, overload

import torch
from torch.distributed.algorithms.join import Joinable, JoinHook
from torch.optim import Optimizer

class _ZeROJoinHook(JoinHook):
    zero: Any = ...
    def __init__(self, zero: Any) -> None: ...
    def main_hook(self) -> None: ...

class _DDPBucketAssignment:
    bucket_index: int
    parameters: list[torch.Tensor]
    offset: int
    device: torch.device
    tensor: torch.Tensor | None

class _OverlapStatus(enum.IntEnum):
    UNINITIALIZED = ...
    DDP_HAS_REBUILT_BUCKETS = ...
    INITIALIZED = ...

class _OverlapInfo:
    status: Any = ...
    params_per_bucket: Any = ...
    params_per_rank: Any = ...
    offsets: Any = ...
    broadcast_handles: Any = ...
    bucket_index_to_future: Any = ...
    bucket_index_to_bucket: Any = ...
    bucket_indices_seen: Any = ...
    assigned_ranks_per_bucket: list[set[int]] = ...
    total_size: int = ...
    shard_buckets: bool = ...
    def __init__(self) -> None: ...
    def wait_for_broadcasts(self) -> None: ...
    def clear_per_iter_info(self) -> None: ...

class ZeroRedundancyOptimizer(Optimizer, Joinable):
    functional_optim_map: Any = ...
    initialized: bool = ...
    process_group: Any = ...
    world_size: int = ...
    rank: int = ...
    global_rank: int = ...
    parameters_as_bucket_view: bool = ...
    optim: Any = ...
    _device_to_device_index: dict[torch.device, int] = ...
    _overlap_with_ddp: bool = ...
    _overlap_info: _OverlapInfo = ...
    _buckets: list[list[torch.Tensor]] = ...
    _bucket_assignments_per_rank: list[dict[int, _DDPBucketAssignment]] = ...
    def __init__(
        self,
        params: Any,
        optimizer_class: type[Optimizer],
        process_group: Any | None = ...,
        parameters_as_bucket_view: bool = ...,
        overlap_with_ddp: bool = ...,
        **defaults: Any,
    ) -> None: ...
    def add_param_group(self, param_group: dict[str, Any]) -> None: ...
    def consolidate_state_dict(self, to: int = ...) -> None: ...
    @overload
    def step(self, closure: None = None, **kwargs: Any) -> None: ...
    @overload
    def step(self, closure: Callable[[], float], **kwargs: Any) -> float: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def _local_step(
        self,
        gradients: list[torch.Tensor | None] | None = None,
        closure: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> float | None: ...
    def _get_assigned_rank(self, bucket_index: int) -> int: ...
    def _init_zero_for_overlap(self) -> None: ...
    def join_hook(self, **kwargs): ...
    @property
    def join_device(self) -> torch.device: ...
    def join_process_group(self) -> Any: ...

```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/distributed/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/optim`, which is part of the **core PyTorch library**.



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

Files in the same folder (`torch/distributed/optim`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`functional_adadelta.py_docs.md`](./functional_adadelta.py_docs.md)
- [`post_localSGD_optimizer.py_docs.md`](./post_localSGD_optimizer.py_docs.md)
- [`functional_adamax.py_docs.md`](./functional_adamax.py_docs.md)
- [`named_optimizer.py_docs.md`](./named_optimizer.py_docs.md)
- [`functional_adagrad.py_docs.md`](./functional_adagrad.py_docs.md)
- [`functional_rprop.py_docs.md`](./functional_rprop.py_docs.md)
- [`functional_adam.py_docs.md`](./functional_adam.py_docs.md)


## Cross-References

- **File Documentation**: `zero_redundancy_optimizer.pyi_docs.md`
- **Keyword Index**: `zero_redundancy_optimizer.pyi_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/optim`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed/optim`):

- [`apply_optimizer_in_backward.py_docs.md_docs.md`](./apply_optimizer_in_backward.py_docs.md_docs.md)
- [`functional_rprop.py_kw.md_docs.md`](./functional_rprop.py_kw.md_docs.md)
- [`functional_adagrad.py_docs.md_docs.md`](./functional_adagrad.py_docs.md_docs.md)
- [`zero_redundancy_optimizer.py_docs.md_docs.md`](./zero_redundancy_optimizer.py_docs.md_docs.md)
- [`_deprecation_warning.py_kw.md_docs.md`](./_deprecation_warning.py_kw.md_docs.md)
- [`zero_redundancy_optimizer.py_kw.md_docs.md`](./zero_redundancy_optimizer.py_kw.md_docs.md)
- [`functional_rmsprop.py_docs.md_docs.md`](./functional_rmsprop.py_docs.md_docs.md)
- [`functional_rprop.py_docs.md_docs.md`](./functional_rprop.py_docs.md_docs.md)
- [`named_optimizer.py_docs.md_docs.md`](./named_optimizer.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `zero_redundancy_optimizer.pyi_docs.md_docs.md`
- **Keyword Index**: `zero_redundancy_optimizer.pyi_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
