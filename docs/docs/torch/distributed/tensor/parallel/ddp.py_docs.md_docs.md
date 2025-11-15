# Documentation: `docs/torch/distributed/tensor/parallel/ddp.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/parallel/ddp.py_docs.md`
- **Size**: 6,478 bytes (6.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/tensor/parallel/ddp.py`

## File Metadata

- **Path**: `torch/distributed/tensor/parallel/ddp.py`
- **Size**: 3,735 bytes (3.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any, Optional

import torch.nn as nn
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _flatten_tensor,
    _unflatten_tensor,
)


__all__ = []  # type: ignore[var-annotated]


def _get_submodule_n_params(module: nn.Module, path: str):
    """
    Get submodule and the direct path of parameter from the module
    """
    if "." in path:
        path_list = path.split(".")
        parent_module_path = ".".join(path_list[:-1])
        module = module.get_submodule(parent_module_path)
        path = path_list[-1]
    return module, path


def _update_module_param(param_list: list[tuple[nn.Module, str, nn.Parameter]]):
    """
    Update parameters within the module
    """
    for item in param_list:
        parent_module, module_path, t = item
        assert hasattr(parent_module, module_path)
        delattr(parent_module, module_path)
        setattr(parent_module, module_path, t)


def _reconstruct_dtensor(module: nn.Module, _input: Any):
    """
    Reconstruct DTensor parameters from local tensors
    """
    param_list = []
    # TODO: To add perf optimizations to this iterations
    for name, t in module.named_parameters():
        if hasattr(t, "_st_info"):
            dtensor = _unflatten_tensor(t, t._st_info)
            param_list.append((*_get_submodule_n_params(module, name), dtensor))
    _update_module_param(param_list)  # type: ignore[arg-type]


def _localize_dtensor(
    module: nn.Module, *_: Any, ignored_params: Optional[set[nn.Parameter]] = None
):
    """
    Convert DTensor parameters to local tensors
    """
    if ignored_params is None:
        ignored_params = set()
    param_list = []
    for name, param in module.named_parameters():
        if param in ignored_params:
            continue
        t, sharding_info = _flatten_tensor(param)
        if sharding_info is not None:
            t = nn.Parameter(t)
            t._st_info = sharding_info  # type: ignore[attr-defined]
            param_list.append((*_get_submodule_n_params(module, name), t))
    _update_module_param(param_list)  # type: ignore[arg-type]


def _pre_dp_module_transform(module: nn.Module):
    """
    Enable the composability between Tensor Parallelism (TP) and Data
    Parallelism(DP) in PyTorch when using DDP. We need to convert Parameters which
    are DTensors to local tensors before wrapping with data parallelism API.
    We then register two hooks, one for converting local tensors back to DTensor
    preforward and one to convert DTensors back to tensors after Forward. By
    integrating this way, we avoid any special handling of DTensor parameters by DDP
    and get DTensor's gradients propagated back to DP, e.g. gradient buckets of DDP.

    For now, this API only works with ``DistributedDataParallel``. It will later support
    other DP methods such as FSDP.

    Args:
        module (:class:`nn.Module`):
            Module which has been applied TP on.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.tensor.parallel.ddp import pre_dp_module_transform
        >>>
        >>> # Define the module.
        >>> m = module(...)
        >>> parallelize_module(m, PairwiseParallel())
        >>> m = pre_dp_module_transform(m)
        >>> m = DDP(m)
        >>>
    """

    _localize_dtensor(module, None, None)
    # TODO: To add test cases and ensure that it works for nested modules
    module.register_forward_pre_hook(_reconstruct_dtensor)
    module.register_forward_hook(_localize_dtensor)

```



## High-Level Overview

"""    Get submodule and the direct path of parameter from the module

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_submodule_n_params`, `_update_module_param`, `_reconstruct_dtensor`, `_localize_dtensor`, `_pre_dp_module_transform`

**Key imports**: Any, Optional, torch.nn as nn, parallelize_module, PairwiseParallel, DistributedDataParallel as DDP, pre_dp_module_transform


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, Optional
- `torch.nn as nn`
- `torch.distributed.tensor.parallel`: parallelize_module, PairwiseParallel
- `torch.nn.parallel`: DistributedDataParallel as DDP
- `torch.distributed.tensor.parallel.ddp`: pre_dp_module_transform


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

Files in the same folder (`torch/distributed/tensor/parallel`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)
- [`loss.py_docs.md`](./loss.py_docs.md)
- [`input_reshard.py_docs.md`](./input_reshard.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`style.py_docs.md`](./style.py_docs.md)
- [`_data_parallel_utils.py_docs.md`](./_data_parallel_utils.py_docs.md)


## Cross-References

- **File Documentation**: `ddp.py_docs.md`
- **Keyword Index**: `ddp.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/tensor/parallel`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/tensor/parallel`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed/tensor/parallel`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`_data_parallel_utils.py_docs.md_docs.md`](./_data_parallel_utils.py_docs.md_docs.md)
- [`fsdp.py_docs.md_docs.md`](./fsdp.py_docs.md_docs.md)
- [`_data_parallel_utils.py_kw.md_docs.md`](./_data_parallel_utils.py_kw.md_docs.md)
- [`loss.py_kw.md_docs.md`](./loss.py_kw.md_docs.md)
- [`style.py_docs.md_docs.md`](./style.py_docs.md_docs.md)
- [`loss.py_docs.md_docs.md`](./loss.py_docs.md_docs.md)
- [`ddp.py_kw.md_docs.md`](./ddp.py_kw.md_docs.md)
- [`style.py_kw.md_docs.md`](./style.py_kw.md_docs.md)
- [`input_reshard.py_docs.md_docs.md`](./input_reshard.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ddp.py_docs.md_docs.md`
- **Keyword Index**: `ddp.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
