# Documentation: `torch/distributed/_functional_collectives_impl.py`

## File Metadata

- **Path**: `torch/distributed/_functional_collectives_impl.py`
- **Size**: 3,269 bytes (3.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional

import torch
import torch.distributed.distributed_c10d as c10d


"""
This file contains the op impls for the legacy (c10d_functional) functional collectives.
These impls simply call into the native (_c10d_functional) functional collectives.
"""


def _broadcast(input, src, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.broadcast(
        input,
        src,
        group_name,
    )


def _all_reduce(input, reduce_op, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_reduce(
        input,
        reduce_op,
        group_name,
    )


def _all_reduce_coalesced(inputs, reduce_op, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_reduce_coalesced(
        inputs,
        reduce_op,
        group_name,
    )


def _all_gather_into_tensor(input, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_gather_into_tensor(
        input,
        group_size,
        group_name,
    )


def _all_gather_into_tensor_coalesced(input, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        input,
        group_size,
        group_name,
    )


def _reduce_scatter_tensor(
    input: torch.Tensor,
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.reduce_scatter_tensor(
        input,
        reduce_op,
        group_size,
        group_name,
    )


def _reduce_scatter_tensor_coalesced(
    inputs: list[torch.Tensor],
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        inputs,
        reduce_op,
        group_size,
        group_name,
    )


def _all_to_all_single(
    input: torch.Tensor,
    output_split_sizes: Optional[list[int]],
    input_split_sizes: Optional[list[int]],
    tag: str,
    ranks: list[int],
    group_size: int,
):
    if output_split_sizes is None or input_split_sizes is None:
        if not (output_split_sizes is None and input_split_sizes is None):
            raise AssertionError(
                "output_split_sizes and input_split_sizes must either be "
                "specified together or both set to None"
            )
        output_split_sizes = [input.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_to_all_single(
        input,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )


def _wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops._c10d_functional.wait_tensor(tensor)

```



## High-Level Overview

"""This file contains the op impls for the legacy (c10d_functional) functional collectives.These impls simply call into the native (_c10d_functional) functional collectives.

This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_broadcast`, `_all_reduce`, `_all_reduce_coalesced`, `_all_gather_into_tensor`, `_all_gather_into_tensor_coalesced`, `_reduce_scatter_tensor`, `_reduce_scatter_tensor_coalesced`, `_all_to_all_single`, `_wait_tensor`

**Key imports**: Optional, torch, torch.distributed.distributed_c10d as c10d


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch.distributed.distributed_c10d as c10d`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_mesh_layout.py_docs.md`](./_mesh_layout.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`c10d_logger.py_docs.md`](./c10d_logger.py_docs.md)
- [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`CONTRIBUTING.md_docs.md`](./CONTRIBUTING.md_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)


## Cross-References

- **File Documentation**: `_functional_collectives_impl.py_docs.md`
- **Keyword Index**: `_functional_collectives_impl.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
