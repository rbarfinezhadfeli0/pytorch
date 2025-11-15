# Documentation: `docs/torch/distributed/tensor/experimental/_context_parallel/_cp_custom_ops.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/experimental/_context_parallel/_cp_custom_ops.py_docs.md`
- **Size**: 5,315 bytes (5.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/tensor/experimental/_context_parallel/_cp_custom_ops.py`

## File Metadata

- **Path**: `torch/distributed/tensor/experimental/_context_parallel/_cp_custom_ops.py`
- **Size**: 2,925 bytes (2.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d


@torch.library.custom_op("cplib::flex_cp_allgather", mutates_args=())
def flex_cp_allgather(
    k: torch.Tensor, v: torch.Tensor, seq_dim: int, pg_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    k = k.contiguous()
    v = v.contiguous()
    k = funcol.all_gather_tensor(k, seq_dim, pg_name)
    v = funcol.all_gather_tensor(v, seq_dim, pg_name)
    if isinstance(k, funcol.AsyncCollectiveTensor):
        k = k.wait()
    if isinstance(v, funcol.AsyncCollectiveTensor):
        v = v.wait()
    return k, v


@flex_cp_allgather.register_fake
def _(
    k: torch.Tensor, v: torch.Tensor, seq_dim: int, pg_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_k = list(k.shape)
    shape_v = list(v.shape)
    shape_k[seq_dim] *= c10d._get_group_size_by_name(pg_name)
    shape_v[seq_dim] *= c10d._get_group_size_by_name(pg_name)
    new_k = torch.empty(shape_k, dtype=k.dtype, device=k.device)
    new_v = torch.empty(shape_v, dtype=v.dtype, device=v.device)
    return new_k, new_v


@torch.library.custom_op("cplib::flex_cp_allgather_backward", mutates_args=())
def flex_cp_allgather_backward(
    grad_full_k: torch.Tensor,
    grad_full_v: torch.Tensor,
    seq_dim: int,
    pg_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_k = funcol.reduce_scatter_tensor(grad_full_k, "sum", seq_dim, pg_name)
    if isinstance(grad_k, funcol.AsyncCollectiveTensor):
        grad_k = grad_k.wait()
    grad_v = funcol.reduce_scatter_tensor(grad_full_v, "sum", seq_dim, pg_name)
    if isinstance(grad_v, funcol.AsyncCollectiveTensor):
        grad_v = grad_v.wait()

    return grad_k, grad_v


@flex_cp_allgather_backward.register_fake
def _(
    grad_full_k: torch.Tensor,
    grad_full_v: torch.Tensor,
    seq_dim: int,
    pg_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_k = list(grad_full_k.shape)
    shape_v = list(grad_full_v.shape)
    shape_k[seq_dim] //= c10d._get_group_size_by_name(pg_name)
    shape_v[seq_dim] //= c10d._get_group_size_by_name(pg_name)
    new_grad_k = torch.empty(
        shape_k, dtype=grad_full_k.dtype, device=grad_full_k.device
    )
    new_grad_v = torch.empty(
        shape_v, dtype=grad_full_v.dtype, device=grad_full_v.device
    )
    return new_grad_k, new_grad_v


def _flex_cp_allgather_backward(
    ctx: Any, grad_full_k: torch.Tensor, grad_full_v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    grad_k, grad_v = flex_cp_allgather_backward(
        grad_full_k, grad_full_v, ctx.seq_dim, ctx.pg_name
    )
    return grad_k, grad_v, None, None


def _flex_cp_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    _, _, ctx.seq_dim, ctx.pg_name = inputs


flex_cp_allgather.register_autograd(
    _flex_cp_allgather_backward, setup_context=_flex_cp_setup_context
)

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `flex_cp_allgather`, `_`, `flex_cp_allgather_backward`, `_`, `_flex_cp_allgather_backward`, `_flex_cp_setup_context`

**Key imports**: Any, torch, torch.distributed._functional_collectives as funcol, torch.distributed.distributed_c10d as c10d


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/experimental/_context_parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch`
- `torch.distributed._functional_collectives as funcol`
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

Files in the same folder (`torch/distributed/tensor/experimental/_context_parallel`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_attention.py_docs.md`](./_attention.py_docs.md)
- [`_load_balancer.py_docs.md`](./_load_balancer.py_docs.md)


## Cross-References

- **File Documentation**: `_cp_custom_ops.py_docs.md`
- **Keyword Index**: `_cp_custom_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/tensor/experimental/_context_parallel`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/tensor/experimental/_context_parallel`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed/tensor/experimental/_context_parallel`):

- [`_attention.py_docs.md_docs.md`](./_attention.py_docs.md_docs.md)
- [`_attention.py_kw.md_docs.md`](./_attention.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`_cp_custom_ops.py_kw.md_docs.md`](./_cp_custom_ops.py_kw.md_docs.md)
- [`_load_balancer.py_kw.md_docs.md`](./_load_balancer.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`_load_balancer.py_docs.md_docs.md`](./_load_balancer.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_cp_custom_ops.py_docs.md_docs.md`
- **Keyword Index**: `_cp_custom_ops.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
