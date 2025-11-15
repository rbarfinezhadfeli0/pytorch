# Documentation: `torch/nn/parallel/replicate.py`

## File Metadata

- **Path**: `torch/nn/parallel/replicate.py`
- **Size**: 6,972 bytes (6.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import cast, TYPE_CHECKING, TypeVar
from typing_extensions import TypeIs

import torch
from torch._utils import _get_device_index
from torch.nn.modules import Module
from torch.nn.parallel import comm


if TYPE_CHECKING:
    from torch._C import ScriptMethod
    from torch.jit import ScriptModule
    from torch.jit._state import EnabledProxy


__all__ = ["replicate"]


def _is_script_module(module: Module) -> TypeIs["ScriptModule"]:
    import torch.jit

    return isinstance(module, torch.jit.ScriptModule)


def _is_script_method(module: object) -> TypeIs["ScriptMethod"]:
    import torch.jit

    return isinstance(module, torch._C.ScriptMethod)


def _init_script_module() -> "ScriptModule":
    import torch.jit

    return torch.jit.ScriptModule()


def _is_jit_enabled() -> "EnabledProxy":
    import torch.jit._state

    return torch.jit._state._enabled


# Check if we can safely replicate the module.
# there are two types of module:
# 1. python modules
# 2. ScriptModule
#
# currently a module cannot be replicated properly if the descendants of
# any ScriptModule contains python module (type 1 above)
def _replicatable_module(module: Module, memo: set[Module] | None = None) -> bool:
    # module.modules() contains module itself as the first element
    def descendant_modules(module: Module) -> Iterator[Module]:
        gen = module.modules()
        next(gen)
        return gen

    if not _is_jit_enabled():
        return True
    if memo is None:
        memo = set()

    # memoize visited modules
    memo.add(module)
    if _is_script_module(module):
        memo.update(descendant_modules(module))
        return all(
            _is_script_module(descendant) for descendant in descendant_modules(module)
        )

    for child in module.children():
        # since any unreplicatable module will cause the check to return
        # False early, visited modules here can be safely ignored.
        if child in memo:
            continue
        if not _replicatable_module(child, memo):
            return False

    return True


def _broadcast_coalesced_reshape(
    tensors: Sequence[torch.Tensor],
    devices: Sequence[int | torch.device],
    detach: bool = False,
) -> list[list[torch.Tensor]]:
    from torch.nn.parallel._functions import Broadcast

    if detach:
        return comm.broadcast_coalesced(tensors, devices)
    else:
        # Use the autograd function to broadcast if not detach
        if len(tensors) > 0:
            tensor_copies = Broadcast.apply(devices, *tensors)
            return [
                tensor_copies[i : i + len(tensors)]
                for i in range(0, len(tensor_copies), len(tensors))
            ]
        else:
            return []


T = TypeVar("T", bound=Module)


def replicate(
    network: T,
    devices: Sequence[int | torch.device],
    detach: bool = False,
) -> list[T]:
    if not _replicatable_module(network):
        raise RuntimeError(
            "Cannot replicate network where python modules are children of ScriptModule"
        )

    if not devices:
        return []

    devices = [_get_device_index(x, True) for x in devices]
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    buffers = list(network.buffers())
    buffers_rg: list[torch.Tensor] = []
    buffers_not_rg: list[torch.Tensor] = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(
        buffers_not_rg, devices, detach=True
    )

    modules = list(network.modules())
    module_copies: list[list[Module]] = [[] for _ in devices]
    module_indices: dict[Module, int] = {}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module._replicate_for_data_parallel()
            # This is a temporary fix for DDP. DDP needs to access the
            # replicated model parameters. It used to do so through
            # `mode.parameters()`. The fix added in #33907 for DP stops the
            # `parameters()` API from exposing the replicated parameters.
            # Hence, we add a `_former_parameters` dict here to support DDP.
            replica._former_parameters = OrderedDict()

            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, module_copies[j][module_idx])
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param_copy = param_copies[j][param_idx]
                    # parameters in replicas are no longer leaves,
                    # so setattr them as non-parameter attributes
                    setattr(replica, key, param_copy)
                    # expose the parameter for DDP
                    replica._former_parameters[key] = param_copy  # type: ignore[operator, index]
        for key, buf in module._buffers.items():  # type: ignore[assignment]
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    return [cast(T, module_copies[j][0]) for j in range(num_replicas)]

```



## High-Level Overview


This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_is_script_module`, `_is_script_method`, `_init_script_module`, `_is_jit_enabled`, `_replicatable_module`, `descendant_modules`, `_broadcast_coalesced_reshape`, `replicate`

**Key imports**: OrderedDict, Iterator, Sequence, cast, TYPE_CHECKING, TypeVar, TypeIs, torch, _get_device_index, Module, comm, ScriptMethod, ScriptModule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: OrderedDict
- `collections.abc`: Iterator, Sequence
- `typing`: cast, TYPE_CHECKING, TypeVar
- `typing_extensions`: TypeIs
- `torch`
- `torch._utils`: _get_device_index
- `torch.nn.modules`: Module
- `torch.nn.parallel`: comm
- `torch._C`: ScriptMethod
- `torch.jit`: ScriptModule
- `torch.jit._state`: EnabledProxy
- `torch.nn.parallel._functions`: Broadcast


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/nn/parallel`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`data_parallel.py_docs.md`](./data_parallel.py_docs.md)
- [`parallel_apply.py_docs.md`](./parallel_apply.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`_functions.py_docs.md`](./_functions.py_docs.md)
- [`scatter_gather.py_docs.md`](./scatter_gather.py_docs.md)
- [`comm.py_docs.md`](./comm.py_docs.md)


## Cross-References

- **File Documentation**: `replicate.py_docs.md`
- **Keyword Index**: `replicate.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
