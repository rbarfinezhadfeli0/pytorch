# Documentation: `torch/distributed/fsdp/_traversal_utils.py`

## File Metadata

- **Path**: `torch/distributed/fsdp/_traversal_utils.py`
- **Size**: 4,610 bytes (4.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
NOTE: This file must be imported like
``import torch.distributed.fsdp._traversal_utils`` and not like
``from torch.distributed.fsdp._traversal_utils import ...`` to avoid circular
imports. For brevity, we may import the file as ``traversal_utils``.
"""

import collections

import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.fsdp._common_utils import _FSDPState, _get_module_fsdp_state


"""
[Note: FSDP State Traversal]
For the wrapper code path, ``_FSDPState`` is the ``FullyShardedDataParallel``
module wrapping a fully sharded module, and for the non-wrapper code path,
``_FSDPState`` is an object that gets embedded on a fully sharded module.
See [Note: Fully Sharded Module] for the definition.

There are three common traversal idioms: Given a root module,
- ``_get_fsdp_states()`` returns all ``_FSDPState`` s in the tree.
- ``get_fsdp_root_states()`` returns all local root ``_FSDPState`` s in the
tree (i.e. those with ``_is_root == True``).
- ``_get_fsdp_handles()``returns all ``FlatParamHandle`` s in the tree.

All of these methods must take in the root module (i.e. an ``nn.Module``) and
not a general ``_FSDPState`` because ``_FSDPState`` does not support a graph
traversal, whereas ``nn.Module`` has ``nn.Module.modules()`` for traversal.
"""


def _composable(module: nn.Module) -> bool:
    """
    Returns if ``module`` can compose with ``fully_shard``.
    """
    # TODO: Add any other composable APIs that are mutually exclusive.
    registry = _get_registry(module)
    if registry is None:
        return True
    return "replicate" not in registry


# TODO (awgu): We may be able to remove this function if we retired the
# `use_orig_params=False` code path since so far we only need the module for
# `FlatParameter` registration, which is not needed for `use_orig_params=True`.
def _get_fsdp_states_with_modules(
    module: nn.Module,
) -> tuple[list[_FSDPState], list[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the modules owning the states in the first list.

    For the wrapper code path, both returned lists are the same, each
    containing all ``FullyShardedDataParallel`` instances. For the composable
    code path, this returns a list of all composable state instances and a list
    of the corresponding fully sharded modules. See [Note: Fully Sharded
    Module].

    NOTE: The traversal does not proceed into any module annotated by an
    incompatible API (e.g. ``replicate``).
    """
    fsdp_states: list[_FSDPState] = []
    fsdp_modules: list[nn.Module] = []
    # Track the visited FSDP states since multiple modules may share the same
    # one and we want to return a de-duplicated list
    visited_fsdp_states: set[_FSDPState] = set()
    # Track the visited modules in case of shared modules, which implies the
    # module graph is no longer a tree
    visited_modules: set[nn.Module] = set()

    # Perform depth-first search from `module` to ensure that we do not
    # traverse into an incompatible API's subtree (use DFS instead of BFS to
    # match `.modules()` order)
    deque: collections.deque[nn.Module] = collections.deque([module])
    while deque:
        submodule = deque.popleft()
        visited_modules.add(submodule)
        if not _composable(submodule):
            continue
        for child_module in reversed(list(submodule.children())):
            if child_module not in visited_modules:
                deque.appendleft(child_module)
        optional_state = _get_module_fsdp_state(submodule)
        if optional_state is not None and optional_state not in visited_fsdp_states:
            visited_fsdp_states.add(optional_state)
            fsdp_states.append(optional_state)
            fsdp_modules.append(submodule)
    return fsdp_states, fsdp_modules


def _get_fsdp_states(module: nn.Module) -> list[_FSDPState]:
    """See :func:`_get_fsdp_states_with_modules`."""
    fsdp_states, _ = _get_fsdp_states_with_modules(module)
    return fsdp_states


def _get_fsdp_handles(module: nn.Module) -> list:
    """
    Returns all ``FlatParamHandle`` s in the module tree rooted at ``module``
    following the rules in :func:`_get_fsdp_state`.
    """
    handles = [
        fsdp_state._handle
        for fsdp_state in _get_fsdp_states(module)
        if fsdp_state._handle is not None
    ]
    return handles

```



## High-Level Overview

"""NOTE: This file must be imported like``import torch.distributed.fsdp._traversal_utils`` and not like``from torch.distributed.fsdp._traversal_utils import ...`` to avoid circularimports. For brevity, we may import the file as ``traversal_utils``.

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_composable`, `_get_fsdp_states_with_modules`, `_get_fsdp_states`, `_get_fsdp_handles`

**Key imports**: torch.distributed.fsdp._traversal_utils, ..., the file as , collections, torch.nn as nn, _get_registry, _FSDPState, _get_module_fsdp_state


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/fsdp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.distributed.fsdp._traversal_utils`
- `...`
- `the file as `
- `collections`
- `torch.nn as nn`
- `torch.distributed._composable.contract`: _get_registry
- `torch.distributed.fsdp._common_utils`: _FSDPState, _get_module_fsdp_state


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

Files in the same folder (`torch/distributed/fsdp`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_limiter_utils.py_docs.md`](./_limiter_utils.py_docs.md)
- [`_runtime_utils.py_docs.md`](./_runtime_utils.py_docs.md)
- [`_common_utils.py_docs.md`](./_common_utils.py_docs.md)
- [`_wrap_utils.py_docs.md`](./_wrap_utils.py_docs.md)
- [`_exec_order_utils.py_docs.md`](./_exec_order_utils.py_docs.md)
- [`sharded_grad_scaler.py_docs.md`](./sharded_grad_scaler.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `_traversal_utils.py_docs.md`
- **Keyword Index**: `_traversal_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
