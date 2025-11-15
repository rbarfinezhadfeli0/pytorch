# Documentation: `torch/distributed/tensor/parallel/input_reshard.py`

## File Metadata

- **Path**: `torch/distributed/tensor/parallel/input_reshard.py`
- **Size**: 3,756 bytes (3.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
from functools import partial
from typing import Any, Optional

import torch
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard


__all__ = [
    "input_reshard",
]


def input_reshard(
    module: torch.nn.Module,
    tp_device_mesh: DeviceMesh,
    input_reshard_dim: Optional[int] = None,
) -> torch.nn.Module:
    """
    Register hooks to an nn.Module for input resharding, enabling sharding and restoration during backward computation.

    Register hooks to an nn.Module with input resharding so that we can shard
    per the given `tp_device_mesh` and `input_reshard_dim` and restore the
    input back when recomputing the activations in the backward. The reason
    why we can do this is that for Tensor Parallel(TP), the input are same
    across all TP ranks.

    Args:
        module (:class:`nn.Module`):
            Module to be registered with input resharding.
        tp_device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for Tensor Parallel.
        input_reshard_dim (Optional[int]):
            The dimension of where we perform the sharding
            of input. If set None, there is no sharding of input.
            Default: None

    Return:
        A :class:`nn.Module` object registered with TP input resharding.
    """
    if input_reshard_dim is None:
        return module

    cx: Optional[torch.autograd.graph.saved_tensors_hooks] = None

    def input_reshard_forward_pre_hook(_: torch.nn.Module, _i: tuple[Any, ...]) -> None:
        saved_tensor_hooks = torch.autograd.graph.saved_tensors_hooks(
            partial(_pack_hook_tp, tp_device_mesh, input_reshard_dim),
            partial(_unpack_hook_tp, tp_device_mesh, input_reshard_dim),
        )
        saved_tensor_hooks.__enter__()
        nonlocal cx
        cx = saved_tensor_hooks  # type: ignore[name-defined]

    def input_reshard_backward_hook(
        _: torch.nn.Module, _i: tuple[Any, ...], _o: Any
    ) -> Any:
        nonlocal cx
        cx.__exit__()  # type: ignore[name-defined, union-attr]

    module.register_forward_pre_hook(input_reshard_forward_pre_hook)
    module.register_forward_hook(input_reshard_backward_hook)
    return module


def _pack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: torch.Tensor) -> Any:  # noqa: D401
    """Hook function called after FWD to shard input."""
    if isinstance(x, DTensor) and all(p.is_replicate() for p in x._spec.placements):
        return x.redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(x, device_mesh=mesh)
            .redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
            .to_local()
        )
    else:
        return x


def _unpack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: Any) -> torch.Tensor:  # noqa: D401
    """Hook function called before activation recomputing in BWD to restore input."""
    if (
        isinstance(x, DTensor)
        and len(x._spec.placements) == 1
        and x._spec.placements[0].is_shard()
    ):
        return x.redistribute(device_mesh=mesh, placements=[Replicate()])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(
                x, device_mesh=mesh, placements=[Shard(input_reshard_dim)]
            )
            .redistribute(device_mesh=mesh, placements=[Replicate()])
            .to_local()
        )
    else:
        return x

```



## High-Level Overview

"""    Register hooks to an nn.Module for input resharding, enabling sharding and restoration during backward computation.    Register hooks to an nn.Module with input resharding so that we can shard    per the given `tp_device_mesh` and `input_reshard_dim` and restore the    input back when recomputing the activations in the backward. The reason    why we can do this is that for Tensor Parallel(TP), the input are same    across all TP ranks.    Args:        module (:class:`nn.Module`):            Module to be registered with input resharding.        tp_device_mesh (:class:`DeviceMesh`):            Object which describes the mesh topology            of devices for Tensor Parallel.        input_reshard_dim (Optional[int]):            The dimension of where we perform the sharding            of input. If set None, there is no sharding of input.            Default: None    Return:        A :class:`nn.Module` object registered with TP input resharding.

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `input_reshard`, `input_reshard_forward_pre_hook`, `input_reshard_backward_hook`, `_pack_hook_tp`, `_unpack_hook_tp`

**Key imports**: partial, Any, Optional, torch, DeviceMesh, DTensor, Replicate, Shard


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`: partial
- `typing`: Any, Optional
- `torch`
- `torch.distributed.tensor`: DeviceMesh, DTensor, Replicate, Shard


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`torch/distributed/tensor/parallel`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)
- [`loss.py_docs.md`](./loss.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`style.py_docs.md`](./style.py_docs.md)
- [`_data_parallel_utils.py_docs.md`](./_data_parallel_utils.py_docs.md)
- [`ddp.py_docs.md`](./ddp.py_docs.md)


## Cross-References

- **File Documentation**: `input_reshard.py_docs.md`
- **Keyword Index**: `input_reshard.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
