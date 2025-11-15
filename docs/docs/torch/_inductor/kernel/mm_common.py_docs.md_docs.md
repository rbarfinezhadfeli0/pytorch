# Documentation: `docs/torch/_inductor/kernel/mm_common.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/kernel/mm_common.py_docs.md`
- **Size**: 11,222 bytes (10.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/kernel/mm_common.py`

## File Metadata

- **Path**: `torch/_inductor/kernel/mm_common.py`
- **Size**: 8,078 bytes (7.89 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch._inductor.select_algorithm import realize_inputs, SymbolicGridFn
from torch._inductor.utils import get_current_backend, sympy_product
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

from .. import config
from ..codegen.wrapper import PythonWrapperCodegen
from ..ir import _IntLike, Layout, TensorBox
from ..utils import load_template


log = logging.getLogger(__name__)


@SymbolicGridFn
def mm_grid(m, n, meta, *, cdiv):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


@SymbolicGridFn
def persistent_mm_grid(M: int, N: int, meta: dict[str, Any], *, cdiv, min):
    """Defines the grid for persistent kernels."""
    return (
        min(meta["NUM_SMS"], cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"])),
        1,
        1,
    )


@SymbolicGridFn
def persistent_grouped_mm_grid(*args):
    meta = args[-1]
    return (meta["NUM_SMS"], 1, 1)


def acc_type(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return "tl.float32"
    return f"tl.{dtype}".replace("torch.", "")


def mm_args(
    mat1,
    mat2,
    *others,
    layout=None,
    out_dtype=None,
    use_4x2_dim=False,
    mat2_transposed=False,
):
    """
    Common arg processing for mm,bmm,addmm,etc
    """
    mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    if mat2_transposed:
        *b2, n, k2 = mat2.get_size()
    else:
        *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.check_equals_and_simplify(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.check_equals_and_simplify(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [*b, m, n],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."
    from ..lowering import expand

    others = [realize_inputs(expand(x, layout.size)) for x in others]

    return [m, n, k, layout, mat1, mat2, *others]


def addmm_epilogue(dtype, alpha, beta):
    def epilogue(acc, bias):
        if alpha != 1:
            acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
        if beta != 1:
            bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
        return V.ops.add(acc, bias)

    return epilogue


def scale_mm_epilogue():
    """
    Create an epilogue function that applies scaling to matrix multiplication result
    using the given scale factors.

    Args:
        dtype: The data type of the output
        scale_a: Scale factor for matrix A
        scale_b: Scale factor for matrix B

    Returns:
        Epilogue function that takes the accumulator and applies scaling
    """

    def epilogue(acc, inv_a_scale, inv_b_scale, bias=None):
        # The epilogue function receives the accumulator (result of mat1 @ mat2)
        # and applies the scaling factors
        # In the original scaled_mm, we use inverse scales, so we multiply by them
        mul_scales = V.ops.mul(inv_a_scale, inv_b_scale)
        mul_acc = V.ops.mul(acc, mul_scales)
        if bias is not None:
            return V.ops.add(mul_acc, bias)
        else:
            return mul_acc

    return epilogue


def use_native_matmul(mat1, mat2):
    if not config.triton.native_matmul:
        return False

    # If tma matmul is on, don't do native matmul
    if (
        config.triton.enable_persistent_tma_matmul
        and torch.utils._triton.has_triton_tma_device()
    ):
        raise AssertionError("native matmul doesn't support tma codegen yet")

    # Currently only enable native matmul for default indexing
    # TODO : support block ptr
    if config.triton.use_block_ptr:
        raise AssertionError("native matmul doesn't support block_ptr codegen yet")

    # Currently only enable native matmul for triton on GPU.
    device_type = mat1.get_device().type
    if not (
        device_type in ("cuda", "xpu") and get_current_backend(device_type) == "triton"
    ):
        return False

    # Currently, tl.dot only supports following dtypes
    triton_supported_dtype = [
        torch.int8,
        torch.uint8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    if mat1.dtype not in triton_supported_dtype:
        return False
    if mat2.dtype not in triton_supported_dtype:
        return False

    # (..., M, K) @ (..., K, N)
    m, k, n = mat1.get_size()[-2], mat1.get_size()[-1], mat2.get_size()[-1]

    # If the shape has unbacked symbols, don't do native matmul.
    # This is related to the behavior of statically_known_multiple_of on unbacked symints.
    # Since statically_known_multiple_of just returns False for unbacked symbols
    # due to the expensive cost, codegen fails when there is a unbacked symbol.
    # In particular, it fails at _split_iteration_ranges in codegen/simd.py.
    # See this : https://github.com/pytorch/pytorch/pull/131649
    if any(map(has_free_unbacked_symbols, [m, k, n])):
        return False

    # Consider the shape (m,k,n) > 1
    # TODO : support when size = 1
    if (
        V.graph.sizevars.statically_known_leq(m, 1)
        or V.graph.sizevars.statically_known_leq(k, 1)
        or V.graph.sizevars.statically_known_leq(n, 1)
    ):
        return False

    return True


def _is_static_problem(layout: Layout) -> tuple[bool, bool]:
    """
    Check if input tensors and output layout have static shapes and non-zero sizes.

    Args:
        layout: Output layout object with a 'size' attribute.

    Returns:
        Tuple[bool, bool]: (is_static, is_nonzero)
            is_static: True if all shapes are statically known
            is_nonzero: True if all dimensions are non-zero
    """
    static_shape = True
    static_size = PythonWrapperCodegen.statically_known_list_of_ints_or_none(
        layout.size
    )
    if static_size is None:
        nonzero = True
        for s in layout.size:
            sz = PythonWrapperCodegen.statically_known_int_or_none(s)
            if sz is not None and sz == 0:
                nonzero = False
                break
        return False, nonzero
    numel = 1
    for dim in static_size:
        numel *= dim
    nonzero = numel > 0
    return static_shape, nonzero


def check_supported_striding(mat_a: TensorBox, mat_b: TensorBox) -> None:
    def is_row_major(stride: Sequence[_IntLike]) -> bool:
        return stride[-1] == 1

    def is_col_major(stride: Sequence[_IntLike]) -> bool:
        return stride[-2] == 1

    def has_zero_dim(size: Sequence[_IntLike]) -> bool:
        return bool(size[0] == 0 or size[1] == 0)

    # Check mat_a (self) stride requirements
    torch._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )

    # Check mat_b stride requirements
    torch._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


def is_batch_stride_largest_or_zero(mat1, mat2, layout) -> bool:
    """
    Checking if the batch stride is the largest in the stride.
    """
    sizes = [mat1.get_size(), mat2.get_size(), layout.size]
    strides = [mat1.get_stride(), mat2.get_stride(), layout.stride]
    for size, stride in zip(sizes, strides):
        assert len(size) == len(stride) == 3, "Expect 3D tensors"
        if stride[0] != 0 and stride[0] != sympy_product(size[1:]):
            return False

    return True


_KERNEL_TEMPLATE_DIR = Path(__file__).parent / "templates"
load_kernel_template = partial(load_template, template_dir=_KERNEL_TEMPLATE_DIR)

```



## High-Level Overview

"""    The CUDA grid size for matmul triton templates.

This Python file contains 0 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `mm_grid`, `persistent_mm_grid`, `persistent_grouped_mm_grid`, `acc_type`, `mm_args`, `addmm_epilogue`, `epilogue`, `scale_mm_epilogue`, `epilogue`, `use_native_matmul`, `_is_static_problem`, `check_supported_striding`, `is_row_major`, `is_col_major`, `has_zero_dim`, `is_batch_stride_largest_or_zero`

**Key imports**: logging, Sequence, partial, Path, Any, torch, realize_inputs, SymbolicGridFn, get_current_backend, sympy_product, V, has_free_unbacked_symbols


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/kernel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `collections.abc`: Sequence
- `functools`: partial
- `pathlib`: Path
- `typing`: Any
- `torch`
- `torch._inductor.select_algorithm`: realize_inputs, SymbolicGridFn
- `torch._inductor.utils`: get_current_backend, sympy_product
- `torch._inductor.virtualized`: V
- `torch.fx.experimental.symbolic_shapes`: has_free_unbacked_symbols
- `..`: config
- `..codegen.wrapper`: PythonWrapperCodegen
- `..ir`: _IntLike, Layout, TensorBox
- `..utils`: load_template
- `torch._inductor.ir`: FixedLayout
- `..lowering`: expand


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

Files in the same folder (`torch/_inductor/kernel`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mm.py_docs.md`](./mm.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`custom_op.py_docs.md`](./custom_op.py_docs.md)
- [`mm_plus_mm.py_docs.md`](./mm_plus_mm.py_docs.md)
- [`bmm.py_docs.md`](./bmm.py_docs.md)
- [`mm_grouped.py_docs.md`](./mm_grouped.py_docs.md)


## Cross-References

- **File Documentation**: `mm_common.py_docs.md`
- **Keyword Index**: `mm_common.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/kernel`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/kernel`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_inductor/kernel`):

- [`mm_grouped.py_kw.md_docs.md`](./mm_grouped.py_kw.md_docs.md)
- [`mm_common.py_kw.md_docs.md`](./mm_common.py_kw.md_docs.md)
- [`mm.py_kw.md_docs.md`](./mm.py_kw.md_docs.md)
- [`bmm.py_kw.md_docs.md`](./bmm.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`bmm.py_docs.md_docs.md`](./bmm.py_docs.md_docs.md)
- [`mm_plus_mm.py_kw.md_docs.md`](./mm_plus_mm.py_kw.md_docs.md)
- [`conv.py_kw.md_docs.md`](./conv.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mm_common.py_docs.md_docs.md`
- **Keyword Index**: `mm_common.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
