# Documentation: `docs/torch/_inductor/inductor_prims.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/inductor_prims.py_docs.md`
- **Size**: 10,149 bytes (9.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/inductor_prims.py`

## File Metadata

- **Path**: `torch/_inductor/inductor_prims.py`
- **Size**: 7,334 bytes (7.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
import operator
from typing import Optional, TYPE_CHECKING

import torch
from torch import _prims, Tensor


if TYPE_CHECKING:
    from collections.abc import Sequence


log = logging.getLogger(__name__)


def make_prim(
    schema: str,
    impl_aten,
    return_type=_prims.RETURN_TYPE.NEW,
    doc: str = "",
    tags: Optional[Sequence[torch.Tag]] = None,
):
    if isinstance(return_type, tuple):

        def meta(*args, **kwargs):
            return tuple(_prims.TensorMeta(o) for o in impl_aten(*args, **kwargs))

    else:

        def meta(*args, **kwargs):
            return _prims.TensorMeta(impl_aten(*args, **kwargs))

    return _prims._make_prim(
        schema=schema,
        return_type=return_type,
        meta=meta,
        impl_aten=impl_aten,
        doc=doc,
        tags=tags,
    )


def eager_force_stride(input_tensor: Tensor, stride) -> Tensor:
    if input_tensor.stride() == stride:
        return input_tensor
    new_tensor = input_tensor.clone().as_strided(
        input_tensor.shape,
        stride,
    )
    new_tensor.copy_(input_tensor)
    return new_tensor


def eager_prepare_softmax(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    amax = torch.amax(x, dim, keepdim=True)
    return amax, torch.sum(torch.exp(x - amax), dim, keepdim=True)


# Custom prims used for handling randomness
seed = make_prim(
    "inductor_seed(Device device) -> Tensor",
    lambda device: torch.randint(2**63 - 1, [], device=device),
    doc="create a fresh seed (one per call) for use with inductor_rand",
    tags=(torch.Tag.nondeterministic_seeded,),
)
seeds = make_prim(
    "inductor_seeds(int count, Device device) -> Tensor",
    lambda count, device: torch.randint(2**63 - 1, [count], device=device),
    doc="Horizontal fusion of many inductor_seed() calls",
    tags=(torch.Tag.nondeterministic_seeded,),
)
lookup_seed = make_prim(
    # if inductor_lookup_seed changes, update partitioners.py
    "inductor_lookup_seed(Tensor seeds, int index) -> Tensor",
    lambda seeds, index: seeds[index].clone(),
    doc="Extract a single seed from the result of inductor_seeds()",
)
# inductor_random() doesn't accept a dtype.
# instead, its lowering always burns in float32, and conversions to a different type
# are explicit in the graph. We therefore need this impl (used during tracing) to hardcoded
# the dtype, so it always faithfully produces a float32 tensor during tracing,
# even if the default dtype is set to something else.
random = make_prim(
    "inductor_random(SymInt[] size, Tensor seed, str mode) -> Tensor",
    lambda size, seed, mode: getattr(torch, mode)(
        size, device=seed.device, dtype=torch.float32
    ),
    doc="torch.rand()/torch.randn() using backend-specific RNG that can be fused",
)
randint = make_prim(
    "inductor_randint(SymInt low, SymInt high, SymInt[] size, Tensor seed) -> Tensor",
    lambda low, high, size, seed: torch.randint(low, high, size, device=seed.device),
    doc="torch.randint() using backend-specific RNG that can be fused",
)
force_stride_order = make_prim(
    "inductor_force_stride_order(Tensor input, SymInt[] stride) -> Tensor",
    eager_force_stride,
    doc="Force the stride order for input tensor. No-op if the input tensor already has the stride. Do a copy otherwise",
)
_unsafe_index_put_ = make_prim(
    "_unsafe_index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
    lambda self, indices, values, accumulate=False: torch.ops.aten.index_put_(
        self, indices, values, accumulate
    ),
    doc="Unsafe index_put_ (doesn't issue device asserts)",
)
fma = make_prim(
    "fma(Tensor a, Tensor b, Tensor c) -> Tensor",
    lambda a, b, c: (a * b) + c,
    doc="Fused multiply add: fma(a, b, c) -> (a * b) + c without rounding after the multiplication",
    tags=(torch.Tag.pointwise,),
)
prepare_softmax_online = make_prim(
    "prepare_softmax_online(Tensor a, int dim) -> (Tensor, Tensor)",
    eager_prepare_softmax,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Prepare the softmax by computing the max and sum.",
)


def _flattened_index_to_nd(indices, width):
    import sympy

    from torch.utils._sympy.functions import FloorDiv

    dim = len(width)

    if dim == 1:
        return [indices]
    elif dim >= 2:
        m = functools.reduce(operator.mul, width[1:])
        if isinstance(indices, sympy.Expr) or isinstance(m, sympy.Expr):
            ih = FloorDiv(indices, m)
        else:
            ih = indices // m
        indices_new = indices - (ih * m)
        return [ih, *_flattened_index_to_nd(indices_new, width[1:])]
    else:
        raise ValueError(f"Unknown dim: {dim}")


def _flatten_index(indices, width):
    result = indices[0]
    for d in range(1, len(indices)):
        result = width[d] * result + indices[d]
    return result


def _low_memory_max_pool_with_offsets_aten(
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    dim = len(kernel_size)
    if dim == 2:
        vals, indices = torch.ops.aten.max_pool2d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )
    else:
        vals, indices = torch.ops.aten.max_pool3d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )

    idhw = _flattened_index_to_nd(indices, self.shape[-dim:])

    dhw_inc = []

    for d in range(dim):
        bh_shape = [1] * self.ndim
        bh_shape[-dim + d] = -1
        bh = torch.arange(
            indices.shape[-dim + d], dtype=torch.int64, device=self.device
        ).view(bh_shape)
        hbase = bh * stride[d] - padding[d]
        h_inc = (idhw[d] - hbase) // dilation[d]
        dhw_inc.append(h_inc)

    offsets = _flatten_index(dhw_inc, kernel_size)

    return vals, offsets.to(torch.int8)


def _low_memory_max_pool_offsets_to_indices_aten(
    offsets,
    kernel_size,
    input_size,
    stride,
    padding,
    dilation,
):
    dim = len(kernel_size)
    offsets = offsets.to(torch.int64)
    dhw_inc = _flattened_index_to_nd(offsets, kernel_size)

    idhw = []
    for d in range(dim):
        bh_shape = [1] * offsets.ndim
        bh_shape[-dim + d] = -1
        bh = torch.arange(
            offsets.shape[-dim + d], dtype=torch.int64, device=offsets.device
        ).view(bh_shape)
        hbase = bh * stride[d] - padding[d]
        idhw.append(hbase + dhw_inc[d] * dilation[d])

    return _flatten_index(idhw, input_size)


_low_memory_max_pool_with_offsets = make_prim(
    "_low_memory_max_pool_with_offsets(Tensor self, SymInt[] kernel_size, SymInt[] stride,  SymInt[] padding, SymInt[] dilation, bool ceil_mode) -> (Tensor, Tensor)",  # noqa: B950
    _low_memory_max_pool_with_offsets_aten,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Instead of returning indices, returns indices offsets.",
)

_low_memory_max_pool_offsets_to_indices = make_prim(
    "_low_memory_max_pool_offsets_to_indices(Tensor self, SymInt[] kernel_size, SymInt[] input_size, SymInt[] stride, SymInt[] padding, SymInt[] dilation) -> Tensor",  # noqa: B950
    _low_memory_max_pool_offsets_to_indices_aten,
    doc="Convert small int offsets to regular indices.",
)

```



## High-Level Overview


This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `make_prim`, `meta`, `meta`, `eager_force_stride`, `eager_prepare_softmax`, `_flattened_index_to_nd`, `_flatten_index`, `_low_memory_max_pool_with_offsets_aten`, `_low_memory_max_pool_offsets_to_indices_aten`

**Key imports**: annotations, functools, logging, operator, Optional, TYPE_CHECKING, torch, _prims, Tensor, Sequence, sympy, FloorDiv


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `functools`
- `logging`
- `operator`
- `typing`: Optional, TYPE_CHECKING
- `torch`
- `collections.abc`: Sequence
- `sympy`
- `torch.utils._sympy.functions`: FloorDiv


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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `inductor_prims.py_docs.md`
- **Keyword Index**: `inductor_prims.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `inductor_prims.py_docs.md_docs.md`
- **Keyword Index**: `inductor_prims.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
