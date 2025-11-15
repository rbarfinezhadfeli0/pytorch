# Documentation: `torch/ao/pruning/sparsifier/utils.py`

## File Metadata

- **Path**: `torch/ao/pruning/sparsifier/utils.py`
- **Size**: 4,997 bytes (4.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from itertools import chain
from typing import Any

from torch import nn
from torch.nn.utils.parametrize import is_parametrized, type_before_parametrizations


__all__ = [
    "module_contains_param",
    "swap_module",
    "module_to_fqn",
    "fqn_to_module",
    "get_arg_info_from_tensor_fqn",
    "FakeSparsity",
]


def module_contains_param(module: nn.Module, parametrization: type[nn.Module]) -> bool:
    if is_parametrized(module):
        # see if any of the module tensors have a parametriztion attached that matches the one passed in
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for key, param_list in module.parametrizations.items()  # type: ignore[union-attr,operator]
        )
    return False


def swap_module(
    mod: nn.Module, mapping: dict[type[nn.Module], type[nn.Module]]
) -> nn.Module:
    r"""Swaps the module using from_dense according to the mapping passed in.
    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to sparse nn module
    Return:
        The corresponding sparse module of `mod` according to mapping, created using from_dense
    """
    if type_before_parametrizations(mod) in mapping:
        sparse_mod = mapping[type_before_parametrizations(mod)]

        # TODO Fix this typing, as Type[Module] has no attribute "from_dense"
        new_mod = sparse_mod.from_dense(mod)  # type: ignore[attr-defined]

        # Preserve module's pre forward hooks. They'll be called on quantized input
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        # Preserve module's post forward hooks except _observer_forward_hook
        # After convert they'll work with quantized output
        for hook_fn in mod._forward_hooks.values():
            new_mod.register_forward_hook(hook_fn)

        # respect device affinity when swapping modules
        # pyrefly: ignore [bad-argument-type]
        devices = {p.device for p in chain(mod.parameters(), mod.buffers())}
        if len(devices) > 1:
            raise AssertionError(
                f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)

        return new_mod

    else:
        return mod


def module_to_fqn(model: nn.Module, module: nn.Module, prefix: str = "") -> str | None:
    """
    Returns the fqn for a module or None if module not a descendent of model.
    """
    if module is model:
        return ""
    for name, child in model.named_children():
        fqn = module_to_fqn(child, module, ".")
        if isinstance(fqn, str):
            return prefix + name + fqn
    return None


def fqn_to_module(model: nn.Module | None, path: str) -> nn.Module | None:
    """
    Given an fqn, returns the corresponding module or tensor or None if the fqn given by `path`
    doesn't correspond to anything. Similar to model.get_submodule(path) but works for tensors.
    """
    if path != "":
        for name in path.split("."):
            model = getattr(model, name, None)
    return model


def get_arg_info_from_tensor_fqn(model: nn.Module, tensor_fqn: str) -> dict[str, Any]:
    """
    Uses tensor_fqn to obtain a dict containing module_fqn, module and tensor_name
    """
    # string manip to split tensor_fqn into module_fqn and tensor_name
    # if tensor_fqn is 'weight' then module_fqn and tensor_name are '' and 'weight'
    # if tensor_fqn is 'linear.weight' then module_fqn and tensor_name are 'linear' and 'weight'
    tensor_name = tensor_fqn.rsplit(".", maxsplit=1)[-1]
    module_fqn = tensor_fqn[: -len(tensor_name) - ("." in tensor_fqn)]

    module = fqn_to_module(model, module_fqn)

    return {
        "module_fqn": module_fqn,
        "module": module,
        "tensor_name": tensor_name,
        "tensor_fqn": tensor_fqn,
    }


# Parametrizations
class FakeSparsity(nn.Module):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        if self.mask.shape != x.shape:
            raise AssertionError(
                f"mask shape ({self.mask.shape}) must match x shape ({x.shape})"
            )
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        # We don't want to let the parametrizations to save the mask.
        # That way we make sure that the linear module doesn't store the masks
        # alongside their parametrizations.
        return {}

```



## High-Level Overview

r"""Swaps the module using from_dense according to the mapping passed in.    Args:        mod: input module        mapping: a dictionary that maps from nn module to sparse nn module    Return:        The corresponding sparse module of `mod` according to mapping, created using from_dense

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FakeSparsity`

**Functions defined**: `module_contains_param`, `swap_module`, `module_to_fqn`, `fqn_to_module`, `get_arg_info_from_tensor_fqn`, `__init__`, `forward`, `state_dict`

**Key imports**: chain, Any, nn, is_parametrized, type_before_parametrizations


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/sparsifier`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`: chain
- `typing`: Any
- `torch`: nn
- `torch.nn.utils.parametrize`: is_parametrized, type_before_parametrizations


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`torch/ao/pruning/sparsifier`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`weight_norm_sparsifier.py_docs.md`](./weight_norm_sparsifier.py_docs.md)
- [`nearly_diagonal_sparsifier.py_docs.md`](./nearly_diagonal_sparsifier.py_docs.md)
- [`base_sparsifier.py_docs.md`](./base_sparsifier.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
