# Documentation: `docs/torch/distributed/fsdp/_fully_shard/_fsdp_init.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/fsdp/_fully_shard/_fsdp_init.py_docs.md`
- **Size**: 12,391 bytes (12.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/fsdp/_fully_shard/_fsdp_init.py`

## File Metadata

- **Path**: `torch/distributed/fsdp/_fully_shard/_fsdp_init.py`
- **Size**: 9,184 bytes (8.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import itertools
import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._logging import warning_once
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor import DeviceMesh, DTensor, init_device_mesh
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from ._fsdp_common import _is_composable_with_fsdp, FSDPMeshInfo, HSDPMeshInfo
from ._fsdp_state import _get_module_fsdp_state


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")


def _get_post_forward_mesh_info(
    reshard_after_forward: Union[bool, int], mesh_info: FSDPMeshInfo
) -> Optional[FSDPMeshInfo]:
    shard_mesh_size = mesh_info.shard_mesh_size
    if not isinstance(reshard_after_forward, (bool, int)):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )
    # NOTE: `isinstance(False, int)` returns `True`.
    if not isinstance(reshard_after_forward, bool) and isinstance(
        reshard_after_forward, int
    ):
        if (
            reshard_after_forward < 1
            or reshard_after_forward > shard_mesh_size
            or shard_mesh_size % reshard_after_forward != 0
        ):
            raise ValueError(
                "If passing reshard_after_forward as an int, it should be a "
                f"factor of {shard_mesh_size}, not {reshard_after_forward}"
            )
        elif reshard_after_forward == 1:
            msg = (
                "reshard_after_forward=1 (int) means resharding parameters to world size 1, "
                "instead of reshard_after_forward=True (bool)"
            )
            warning_once(logger, msg, stacklevel=2)
            reshard_after_forward = False
        elif reshard_after_forward == shard_mesh_size:
            reshard_after_forward = True
    post_forward_mesh_info = None
    if reshard_after_forward is True:
        post_forward_mesh_info = mesh_info
    elif reshard_after_forward is not False:  # int case
        # For HSDP, we can flatten the two replicate dims into the 0th dim
        post_forward_mesh_tensor = mesh_info.mesh.mesh.view(-1, reshard_after_forward)
        post_forward_mesh = DeviceMesh(
            mesh_info.mesh.device_type, post_forward_mesh_tensor
        )
        post_forward_mesh_info = HSDPMeshInfo(
            post_forward_mesh, shard_mesh_dim=1, replicate_mesh_dim=0
        )
    return post_forward_mesh_info


def _init_default_fully_shard_mesh() -> DeviceMesh:
    """Default to global CUDA mesh if possible else global CPU mesh."""
    if not dist.distributed_c10d.is_initialized():
        dist.distributed_c10d.init_process_group()
    default_pg = dist.distributed_c10d._get_default_group()
    device = torch._C._get_accelerator()
    mesh = init_device_mesh(device.type, mesh_shape=(default_pg.size(),))
    return mesh


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())


def _ignore_module(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    ignore_decision: dict[nn.Module, bool],
) -> bool:
    """
    Decide if it is safe to ignore a module for applying fully_shard.
    """
    if module in ignore_decision:
        return ignore_decision[module]

    if len(list(module.buffers(recurse=False))) > 0:
        # Cannot ignore a module with any buffer
        ignore_decision[module] = False
        return False

    for _, param in module.named_parameters(recurse=False):
        if param not in ignored_params:
            # at least one param is not ignored. So this module shouldn't be.
            ignore_decision[module] = False
            return False

    # Need to consider descendants of module
    for child in list(module.children()):
        ignore_child = _ignore_module(child, ignored_params, ignore_decision)
        if not ignore_child:
            # Cannot ignore module if one of its children is not ignored
            ignore_decision[module] = False
            return False

    # Safe to ignore module
    ignore_decision[module] = True
    return True


def _adjust_managed_modules(
    modules: list[nn.Module], ignored_params: set[nn.Parameter]
) -> list[nn.Module]:
    """
    Adjust the given list of managed modules by removing those with all parameters ignored.
    """
    ignore_decision: dict[nn.Module, bool] = {}
    new_modules = []
    for module in modules:
        ignored = _ignore_module(module, ignored_params, ignore_decision)
        if not ignored:
            new_modules.append(module)
    return new_modules


def _get_managed_modules(
    root_modules: tuple[nn.Module, ...],
    ignored_params: Optional[set[nn.Parameter]] = None,
) -> list[nn.Module]:
    modules: list[nn.Module] = []
    root_modules_set = set(root_modules)
    # Track visisted modules to avoid visiting shared modules multiple times
    visited_modules: set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        """
        Runs a DFS to collect managed modules, not recursing into modules with
        a non-composable API or ``fully_shard`` already applied.
        """
        if not _is_composable_with_fsdp(module):
            return
        elif (
            module not in root_modules_set
            and _get_module_fsdp_state(module) is not None
        ):
            return  # nested `fully_shard` module
        visited_modules.add(module)
        for submodule in module.children():
            if submodule not in visited_modules:
                dfs(submodule)
        modules.append(module)

    for root_module in root_modules:
        dfs(root_module)

    if ignored_params is None:
        return modules

    adjusted_modules = _adjust_managed_modules(modules, ignored_params)
    return adjusted_modules


def _verify_managed_param(name: str, param: nn.Parameter) -> None:
    """
    Verify if the parameter is accepted by fully_shard. The only restriction now
    is that the parameter cannot be a scalar tensor (param.numel == 0) since we
    need at least one dim to shard.
    """
    if len(param.shape) == 0:
        raise ValueError(
            "fully_shard doesn't support scalar parameters. "
            f"Change {name} to a 1D tensor with numel equal to 1."
        )


def _get_managed_states(
    modules: list[nn.Module], ignored_params: Optional[set[nn.Parameter]] = None
) -> tuple[list[nn.Parameter], list[torch.Tensor]]:
    params: list[nn.Parameter] = []
    buffers: list[torch.Tensor] = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params: set[nn.Parameter] = set()
    visited_buffers: set[torch.Tensor] = set()
    if ignored_params is None:
        ignored_params = set()

    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if param in ignored_params:
                # do not include an ignored parameters
                continue
            if param not in visited_params:
                _verify_managed_param(name, param)
                params.append(param)
                visited_params.add(param)
        for buffer in module.buffers(recurse=False):
            if buffer not in visited_buffers:
                buffers.append(buffer)
                visited_buffers.add(buffer)
    return params, buffers


def _move_states_to_device(
    params: list[nn.Parameter],
    buffers: list[torch.Tensor],
    device: torch.device,
) -> None:
    """
    We have FSDP move states to device for simpler and faster initialization
    since FSDP almost always uses CUDA for training. We move parameters/buffers
    rather than modules since modules to support ignoring parameters/buffers in
    the future.
    """
    # Follow the logic in `nn.Module._apply`
    # pyrefly: ignore [bad-argument-type]
    for tensor in itertools.chain(params, buffers):
        if tensor.device == device or tensor.device.type == "meta":
            # Keep meta-device tensors on meta device for deferred init
            continue
        if isinstance(tensor, DTensor):
            if (dtensor_mesh_type := tensor.device_mesh.device_type) != device.type:
                raise ValueError(
                    "Requires DTensor to have mesh of the same type as the FSDP mesh "
                    f"but got {dtensor_mesh_type} for DTensor and {device.type} for FSDP"
                )
            raise AssertionError(
                f"Expects DTensor to be moved to {dtensor_mesh_type} but got {tensor.device}"
            )
        tensor_ = tensor
        if is_traceable_wrapper_subclass(tensor_):
            with torch.no_grad():  # avoid autograd increasing C++ refcount by 1
                tensor_on_device = nn.Parameter(tensor.to(device))
            torch.utils.swap_tensors(tensor, tensor_on_device)
        else:
            tensor.data = tensor.to(device)

```



## High-Level Overview


This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_post_forward_mesh_info`, `_init_default_fully_shard_mesh`, `_get_device_from_mesh`, `_ignore_module`, `_adjust_managed_modules`, `_get_managed_modules`, `dfs`, `_verify_managed_param`, `_get_managed_states`, `_move_states_to_device`

**Key imports**: itertools, logging, Optional, Union, torch, torch.distributed as dist, torch.nn as nn, warning_once, _get_device_handle, DeviceMesh, DTensor, init_device_mesh, is_traceable_wrapper_subclass


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/fsdp/_fully_shard`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `logging`
- `typing`: Optional, Union
- `torch`
- `torch.distributed as dist`
- `torch.nn as nn`
- `torch._logging`: warning_once
- `torch.distributed.device_mesh`: _get_device_handle
- `torch.distributed.tensor`: DeviceMesh, DTensor, init_device_mesh
- `torch.utils._python_dispatch`: is_traceable_wrapper_subclass
- `._fsdp_common`: _is_composable_with_fsdp, FSDPMeshInfo, HSDPMeshInfo
- `._fsdp_state`: _get_module_fsdp_state


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/distributed/fsdp/_fully_shard`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_fsdp_state.py_docs.md`](./_fsdp_state.py_docs.md)
- [`_fsdp_api.py_docs.md`](./_fsdp_api.py_docs.md)
- [`_fsdp_common.py_docs.md`](./_fsdp_common.py_docs.md)
- [`_fsdp_param_group.py_docs.md`](./_fsdp_param_group.py_docs.md)
- [`_fsdp_collectives.py_docs.md`](./_fsdp_collectives.py_docs.md)
- [`_fsdp_param.py_docs.md`](./_fsdp_param.py_docs.md)
- [`_fully_shard.py_docs.md`](./_fully_shard.py_docs.md)


## Cross-References

- **File Documentation**: `_fsdp_init.py_docs.md`
- **Keyword Index**: `_fsdp_init.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/fsdp/_fully_shard`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/fsdp/_fully_shard`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/distributed/fsdp/_fully_shard`):

- [`_fsdp_common.py_docs.md_docs.md`](./_fsdp_common.py_docs.md_docs.md)
- [`_fsdp_collectives.py_kw.md_docs.md`](./_fsdp_collectives.py_kw.md_docs.md)
- [`_fsdp_param.py_kw.md_docs.md`](./_fsdp_param.py_kw.md_docs.md)
- [`_fsdp_state.py_kw.md_docs.md`](./_fsdp_state.py_kw.md_docs.md)
- [`_fsdp_collectives.py_docs.md_docs.md`](./_fsdp_collectives.py_docs.md_docs.md)
- [`_fully_shard.py_docs.md_docs.md`](./_fully_shard.py_docs.md_docs.md)
- [`_fsdp_init.py_kw.md_docs.md`](./_fsdp_init.py_kw.md_docs.md)
- [`_fsdp_api.py_docs.md_docs.md`](./_fsdp_api.py_docs.md_docs.md)
- [`_fsdp_state.py_docs.md_docs.md`](./_fsdp_state.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_fsdp_init.py_docs.md_docs.md`
- **Keyword Index**: `_fsdp_init.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
