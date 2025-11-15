# Documentation: `torch/utils/_foreach_utils.py`

## File Metadata

- **Path**: `torch/utils/_foreach_utils.py`
- **Size**: 2,409 bytes (2.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import TypeAlias

import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad


def _get_foreach_kernels_supported_devices() -> list[str]:
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda", "xpu", "mtia", torch._C._get_privateuse1_backend_name()]


def _get_fused_kernels_supported_devices() -> list[str]:
    r"""Return the device type list that supports fused kernels in optimizer."""
    return [
        "mps",
        "cuda",
        "xpu",
        "hpu",
        "cpu",
        "mtia",
        torch._C._get_privateuse1_backend_name(),
    ]


TensorListList: TypeAlias = list[list[Tensor | None]]
Indices: TypeAlias = list[int]
_foreach_supported_types = [torch.Tensor]


# This util function splits tensors into groups by device and dtype, which is useful before sending
# tensors off to a foreach implementation, which requires tensors to be on one device and dtype.
# If tensorlistlist contains more than one tensorlist, the following assumptions are made BUT NOT verified:
#   - tensorlists CAN be None
#   - all tensors in the first specified list cannot be None
#   - given an index i, all specified tensorlist[i]s match in dtype and device
# with_indices (bool, optional): whether to track previous indices as the last list per dictionary entry.
#   It comes in handy if there are Nones or literals in the tensorlists that are getting scattered out.
#   Whereas mutating a tensor in the resulting split-up tensorlists WILL propagate changes back to the
#   original input tensorlists, changing up Nones/literals WILL NOT propagate, and manual propagation
#   may be necessary. Check out torch/optim/sgd.py for an example.
@no_grad()
def _group_tensors_by_device_and_dtype(
    tensorlistlist: TensorListList,
    with_indices: bool = False,
) -> dict[tuple[torch.device, torch.dtype], tuple[TensorListList, Indices]]:
    return torch._C._group_tensors_by_device_and_dtype(tensorlistlist, with_indices)


def _device_has_foreach_support(device: torch.device) -> bool:
    return (
        device.type in (_get_foreach_kernels_supported_devices() + ["cpu"])
        and not torch.jit.is_scripting()
    )


def _has_foreach_support(tensors: list[Tensor], device: torch.device) -> bool:
    return _device_has_foreach_support(device) and all(
        t is None or type(t) in _foreach_supported_types for t in tensors
    )

```



## High-Level Overview

r"""Return the device type list that supports foreach kernels."""    return ["cuda", "xpu", "mtia", torch._C._get_privateuse1_backend_name()]def _get_fused_kernels_supported_devices() -> list[str]:

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_foreach_kernels_supported_devices`, `_get_fused_kernels_supported_devices`, `_group_tensors_by_device_and_dtype`, `_device_has_foreach_support`, `_has_foreach_support`

**Key imports**: TypeAlias, torch, Tensor, no_grad


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: TypeAlias
- `torch`
- `torch.autograd.grad_mode`: no_grad


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `_foreach_utils.py_docs.md`
- **Keyword Index**: `_foreach_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
