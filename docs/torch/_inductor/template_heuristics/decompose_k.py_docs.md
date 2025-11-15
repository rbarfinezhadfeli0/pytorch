# Documentation: `torch/_inductor/template_heuristics/decompose_k.py`

## File Metadata

- **Path**: `torch/_inductor/template_heuristics/decompose_k.py`
- **Size**: 2,382 bytes (2.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

import torch

from ..ir import get_free_symbols
from ..kernel.mm import decompose_k_subgraph_template
from ..kernel_inputs import KernelInputs, MMKernelInputs
from ..utils import get_k_splits
from ..virtualized import V
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(decompose_k_subgraph_template.uid, None, op_name="mm")
class EmptyDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    """empty heuristics to skip decompose k on anything not cuda"""


# on CUDA, we don't support hip for decompose_k yet
@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="mm",
)
# TODO(coconutruben): enable decompose k on AMD by removing the register bool
# and benchmarking it for performance and stability
# TODO(coconutruben): enable decompose k on other devices (xpu, cpu, mps, mtia)
# by either adding specific register_template_heuristic tags, or setting the
# device to None (enabled on all devices)
class DecomposeKConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get all the valid k_splits for the given m, n, k.
        """
        assert isinstance(kernel_inputs, MMKernelInputs), (
            f"{self.__class__.__name__} requires MMKernelInputs"
        )

        # Check for unbacked symbols - if found, yield nothing
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        )
        if unbacked_symbols:
            return

        m, n, k = kernel_inputs.mnk_symbolic()
        k_splits = get_k_splits(m, n, k)
        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}

```



## High-Level Overview

"""empty heuristics to skip decompose k on anything not cuda"""# on CUDA, we don't support hip for decompose_k yet@register_template_heuristic(    decompose_k_subgraph_template.uid,    "cuda",    register=torch.version.hip is None,    op_name="mm",)# TODO(coconutruben): enable decompose k on AMD by removing the register bool# and benchmarking it for performance and stability# TODO(coconutruben): enable decompose k on other devices (xpu, cpu, mps, mtia)# by either adding specific register_template_heuristic tags, or setting the# device to None (enabled on all devices)class DecomposeKConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):    def _get_template_configs_impl(        self,        kernel_inputs: KernelInputs,        op_name: str,    ) -> Generator[dict[str, Any], None, None]:

This Python file contains 2 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EmptyDecomposeKConfigHeuristics`, `DecomposeKConfigHeuristics`

**Functions defined**: `_get_template_configs_impl`

**Key imports**: annotations, Any, TYPE_CHECKING, sympy, torch, get_free_symbols, decompose_k_subgraph_template, KernelInputs, MMKernelInputs, get_k_splits, V, TemplateConfigHeuristics


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/template_heuristics`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any, TYPE_CHECKING
- `sympy`
- `torch`
- `..ir`: get_free_symbols
- `..kernel.mm`: decompose_k_subgraph_template
- `..kernel_inputs`: KernelInputs, MMKernelInputs
- `..utils`: get_k_splits
- `..virtualized`: V
- `.base`: TemplateConfigHeuristics
- `.gemm`: GemmMaxAutotuneTemplateConfigHeuristics
- `.registry`: register_template_heuristic
- `collections.abc`: Generator


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

Files in the same folder (`torch/_inductor/template_heuristics`):

- [`aten.py_docs.md`](./aten.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`params.py_docs.md`](./params.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`cutedsl.py_docs.md`](./cutedsl.py_docs.md)
- [`base.py_docs.md`](./base.py_docs.md)
- [`contiguous_mm.py_docs.md`](./contiguous_mm.py_docs.md)
- [`triton.py_docs.md`](./triton.py_docs.md)
- [`triton_addmm.py_docs.md`](./triton_addmm.py_docs.md)


## Cross-References

- **File Documentation**: `decompose_k.py_docs.md`
- **Keyword Index**: `decompose_k.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
