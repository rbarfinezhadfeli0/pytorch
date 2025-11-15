# Documentation: `docs/torch/distributed/tensor/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/__init__.py_docs.md`
- **Size**: 4,711 bytes (4.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/distributed/tensor/__init__.py`

## File Metadata

- **Path**: `torch/distributed/tensor/__init__.py`
- **Size**: 2,249 bytes (2.20 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.distributed.tensor._ops  # force import all built-in dtensor ops
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh  # noqa: F401
from torch.distributed.tensor._api import (
    distribute_module,
    distribute_tensor,
    DTensor,
    empty,
    full,
    ones,
    rand,
    randn,
    zeros,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)
from torch.utils._foreach_utils import (
    _foreach_supported_types as _util_foreach_supported_types,
)


# All public APIs from dtensor package
__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
    "Partial",
    "Placement",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "zeros",
]

# For weights_only torch.load
from ._dtensor_spec import (
    DTensorSpec as _DTensorSpec,
    ShardOrderEntry as _ShardOrderEntry,
    TensorMeta as _TensorMeta,
)


torch.serialization.add_safe_globals(
    [
        DeviceMesh,
        _DTensorSpec,
        _TensorMeta,
        _ShardOrderEntry,
        DTensor,
        Partial,
        Replicate,
        Shard,
    ]
)


# Append DTensor to the list of supported types for foreach implementation for optimizer
# and clip_grad_norm_ so that we will try to use foreach over the for-loop implementation on CUDA.
if DTensor not in _optim_foreach_supported_types:
    _optim_foreach_supported_types.append(DTensor)

if DTensor not in _util_foreach_supported_types:
    _util_foreach_supported_types.append(DTensor)  # type: ignore[arg-type]


# Set namespace for exposed private names
DTensor.__module__ = "torch.distributed.tensor"
distribute_tensor.__module__ = "torch.distributed.tensor"
distribute_module.__module__ = "torch.distributed.tensor"
ones.__module__ = "torch.distributed.tensor"
empty.__module__ = "torch.distributed.tensor"
full.__module__ = "torch.distributed.tensor"
rand.__module__ = "torch.distributed.tensor"
randn.__module__ = "torch.distributed.tensor"
zeros.__module__ = "torch.distributed.tensor"

```



## High-Level Overview


This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: torch, torch.distributed.tensor._ops  , all built, DeviceMesh, init_device_mesh  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.tensor._ops  `
- `all built`
- `torch.distributed.device_mesh`: DeviceMesh, init_device_mesh  


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/distributed/tensor`):

- [`_tp_conv.py_docs.md`](./_tp_conv.py_docs.md)
- [`_dispatch.py_docs.md`](./_dispatch.py_docs.md)
- [`_random.py_docs.md`](./_random.py_docs.md)
- [`_collective_utils.py_docs.md`](./_collective_utils.py_docs.md)
- [`device_mesh.py_docs.md`](./device_mesh.py_docs.md)
- [`_redistribute.py_docs.md`](./_redistribute.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`_sharding_prop.py_docs.md`](./_sharding_prop.py_docs.md)
- [`_shards_wrapper.py_docs.md`](./_shards_wrapper.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/distributed/tensor`):

- [`device_mesh.py_docs.md_docs.md`](./device_mesh.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`placement_types.py_kw.md_docs.md`](./placement_types.py_kw.md_docs.md)
- [`device_mesh.py_kw.md_docs.md`](./device_mesh.py_kw.md_docs.md)
- [`_sharding_prop.py_kw.md_docs.md`](./_sharding_prop.py_kw.md_docs.md)
- [`_random.py_kw.md_docs.md`](./_random.py_kw.md_docs.md)
- [`_collective_utils.py_kw.md_docs.md`](./_collective_utils.py_kw.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_collective_utils.py_docs.md_docs.md`](./_collective_utils.py_docs.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
