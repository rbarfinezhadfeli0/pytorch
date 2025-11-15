# Documentation: `docs/torch/_inductor/kernel/mm_plus_mm.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/kernel/mm_plus_mm.py_docs.md`
- **Size**: 8,922 bytes (8.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/kernel/mm_plus_mm.py`

## File Metadata

- **Path**: `torch/_inductor/kernel/mm_plus_mm.py`
- **Size**: 5,897 bytes (5.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import logging
from typing import TYPE_CHECKING, Union

import torch

from .. import config as inductor_config
from ..kernel_inputs import MMKernelInputs
from ..lowering import lowerings
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import mm_args, mm_grid


if TYPE_CHECKING:
    from torch._inductor.ir import ChoiceCaller
    from torch._inductor.select_algorithm import KernelTemplate

log = logging.getLogger(__name__)

aten = torch.ops.aten

aten_mm_plus_mm = ExternKernelChoice(
    torch.ops.inductor._mm_plus_mm, "torch::inductor::_mm_plus_mm"
)

mm_plus_mm_template = TritonTemplate(
    name="mm_plus_mm",
    grid=mm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C", "D")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K1 = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    # K2 = {{size("C", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    stride_cm = {{stride("C", 0)}}
    stride_ck = {{stride("C", 1)}}
    stride_dk = {{stride("D", 0)}}
    stride_dn = {{stride("D", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if (((stride_am == 1 and stride_ak == M) or (stride_am == K1 and stride_ak == 1))
        and ((stride_cm == 1 and stride_ck == M) or (stride_cm == K1 and stride_ck == 1))):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M

    if (((stride_bk == 1 and stride_bn == K1) or (stride_bk == N and stride_bn == 1))
        and ((stride_dk == 1 and stride_dn == K1) or (stride_dk == N and stride_dn == 1))):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    C = C + (ram[:, None] * stride_cm + rk[None, :] * stride_ck)
    D = D + (rk[:, None] * stride_dk + rbn[None, :] * stride_dn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k1 in range(K1, 0, -BLOCK_K):
        # First matmul with A @ B
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k1, other=0.)
            b = tl.load(B, mask=rk[:, None] < k1, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    for k2 in range(K1, 0, -BLOCK_K):

        # Second matmul with C @ D
        if EVEN_K:
            c = tl.load(C)
            d = tl.load(D)
        else:
            c = tl.load(C, mask=rk[None, :] < k2, other=0.)
            d = tl.load(D, mask=rk[:, None] < k2, other=0.)
        acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)
        C += BLOCK_K * stride_ck
        D += BLOCK_K * stride_dk


    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
""",
    cache_codegen_enabled_for_template=True,
)


def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    """
    Computes mm(mat1, mat2) + mm(mat3, mat4)
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m1, n1, k1, layout1, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    m2, n2, _, layout2, mat3, mat4 = mm_args(mat3, mat4, layout=layout)

    # Optimization is optional, because we can always just not do the fusion
    if (
        m1 * n1 == 0
        or m2 * n2 == 0
        or not V.graph.sizevars.statically_known_list_equals(
            mat1.get_size(), mat3.get_size()
        )
        or not V.graph.sizevars.statically_known_list_equals(
            mat2.get_size(), mat4.get_size()
        )
        or inductor_config.triton.native_matmul
    ):
        # TODO(jansel): support different K values when this is fixed:
        # https://github.com/triton-lang/triton/issues/967
        return lowerings[aten.add](
            lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4)
        )

    # Create MMKernelInputs for MM Plus MM (matrices are at indices 0, 1 for first pair)
    # Note: This is a special case with 4 matrices, but we use the first pair for M, N, K extraction
    kernel_inputs = MMKernelInputs([mat1, mat2, mat3, mat4], mat1_idx=0, mat2_idx=1)

    assert layout1 == layout2
    # options to tune from
    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[Union[ExternKernelChoice, KernelTemplate]] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_mm_plus_mm)

    if use_triton_template(layout1, check_max_autotune=False):
        templates_to_use.append(mm_plus_mm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, "mm_plus_mm")
    )

    return autotune_select_algorithm(
        "mm_plus_mm", choices, kernel_inputs.nodes(), layout1
    )

```



## High-Level Overview

source=r"""{{def_kernel("A", "B", "C", "D")}}    M = {{size("A", 0)}}    N = {{size("B", 1)}}    K1 = {{size("A", 1)}}    if M * N == 0:        # early exit due to zero-size input(s)        return    # K2 = {{size("C", 1)}}    stride_am = {{stride("A", 0)}}    stride_ak = {{stride("A", 1)}}    stride_bk = {{stride("B", 0)}}    stride_bn = {{stride("B", 1)}}    stride_cm = {{stride("C", 0)}}

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `tuned_mm_plus_mm`

**Key imports**: logging, TYPE_CHECKING, Union, torch, config as inductor_config, MMKernelInputs, lowerings, use_aten_gemm_kernels, use_triton_template, V, mm_args, mm_grid, ChoiceCaller


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/kernel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `typing`: TYPE_CHECKING, Union
- `torch`
- `..`: config as inductor_config
- `..kernel_inputs`: MMKernelInputs
- `..lowering`: lowerings
- `..utils`: use_aten_gemm_kernels, use_triton_template
- `..virtualized`: V
- `.mm_common`: mm_args, mm_grid
- `torch._inductor.ir`: ChoiceCaller
- `torch._inductor.select_algorithm`: KernelTemplate


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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
- [`mm_common.py_docs.md`](./mm_common.py_docs.md)
- [`mm.py_docs.md`](./mm.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`custom_op.py_docs.md`](./custom_op.py_docs.md)
- [`bmm.py_docs.md`](./bmm.py_docs.md)
- [`mm_grouped.py_docs.md`](./mm_grouped.py_docs.md)


## Cross-References

- **File Documentation**: `mm_plus_mm.py_docs.md`
- **Keyword Index**: `mm_plus_mm.py_kw.md`
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

- Implements or uses **caching** mechanisms.
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
- [`mm_common.py_docs.md_docs.md`](./mm_common.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`bmm.py_docs.md_docs.md`](./bmm.py_docs.md_docs.md)
- [`mm_plus_mm.py_kw.md_docs.md`](./mm_plus_mm.py_kw.md_docs.md)
- [`conv.py_kw.md_docs.md`](./conv.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mm_plus_mm.py_docs.md_docs.md`
- **Keyword Index**: `mm_plus_mm.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
