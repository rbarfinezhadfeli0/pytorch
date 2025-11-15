# Documentation: `docs/torch/_inductor/template_heuristics/aten.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/template_heuristics/aten.py_docs.md`
- **Size**: 6,182 bytes (6.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/template_heuristics/aten.py`

## File Metadata

- **Path**: `torch/_inductor/template_heuristics/aten.py`
- **Size**: 2,879 bytes (2.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch._inductor import config as inductor_config

from ..kernel.bmm import aten_baddbmm, aten_bmm, aten_bmm_dtype
from ..kernel.mm import (
    aten__fp8_mm,
    aten__int_mm,
    aten_addmm,
    aten_bias_addmm,
    aten_mm,
    aten_mm_dtype,
)
from ..kernel.mm_plus_mm import aten_mm_plus_mm
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..kernel_inputs import KernelInputs


# These are all labeled as device type None to indicate that they
# are valid for all device types
@register_template_heuristic(aten_mm.uid, None)
@register_template_heuristic(aten_mm_dtype.uid, "cuda")
@register_template_heuristic(aten__fp8_mm.uid, None)
@register_template_heuristic(aten__int_mm.uid, None)
@register_template_heuristic(aten_bmm.uid, None)
@register_template_heuristic(aten_mm_plus_mm.uid, None)
# bmm dtype is only valid on cuda
@register_template_heuristic(aten_bmm_dtype.uid, "cuda")
class ATenConfigHeuristics(TemplateConfigHeuristics):
    """
    Pseudo heuristic to make ATen choices go through the same flow as other templates

    This is a single choice without kwargs

    If you want to use this with an ATen choice that has kwargs, just subclass
    """

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        yield dict()


# None here indicates that this is valid for all device types on that op
# Note (None, op) takes precedence over (device_type, None)
@register_template_heuristic(aten_addmm.uid, None, op_name="addmm")
@register_template_heuristic(aten_baddbmm.uid, None, op_name="baddbmm")
class ATenAddMMConfigHeuristics(ATenConfigHeuristics):
    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {
            **kwargs,
            "alpha": alpha,
            "beta": beta,
        }


@register_template_heuristic(aten_bias_addmm.uid, None, op_name="addmm")
class ATenBiasAddMMConfigHeuristics(
    ATenAddMMConfigHeuristics, GemmMaxAutotuneTemplateConfigHeuristics
):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        nodes = kernel_inputs.nodes()
        # for addmm, bias is the first input
        bias = nodes[0]
        if bias.get_stride()[0] == 0 and inductor_config.triton.autotune_cublasLt:
            yield dict()

```



## High-Level Overview

"""    Pseudo heuristic to make ATen choices go through the same flow as other templates    This is a single choice without kwargs    If you want to use this with an ATen choice that has kwargs, just subclass

This Python file contains 3 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ATenConfigHeuristics`, `ATenAddMMConfigHeuristics`, `ATenBiasAddMMConfigHeuristics`

**Functions defined**: `_get_template_configs_impl`, `get_extra_kwargs`, `_get_template_configs_impl`

**Key imports**: annotations, Any, TYPE_CHECKING, config as inductor_config, aten_baddbmm, aten_bmm, aten_bmm_dtype, aten_mm_plus_mm, TemplateConfigHeuristics, GemmMaxAutotuneTemplateConfigHeuristics, register_template_heuristic, Generator, KernelInputs


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/template_heuristics`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any, TYPE_CHECKING
- `torch._inductor`: config as inductor_config
- `..kernel.bmm`: aten_baddbmm, aten_bmm, aten_bmm_dtype
- `..kernel.mm_plus_mm`: aten_mm_plus_mm
- `.base`: TemplateConfigHeuristics
- `.gemm`: GemmMaxAutotuneTemplateConfigHeuristics
- `.registry`: register_template_heuristic
- `collections.abc`: Generator
- `..kernel_inputs`: KernelInputs


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

Files in the same folder (`torch/_inductor/template_heuristics`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`params.py_docs.md`](./params.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`cutedsl.py_docs.md`](./cutedsl.py_docs.md)
- [`decompose_k.py_docs.md`](./decompose_k.py_docs.md)
- [`base.py_docs.md`](./base.py_docs.md)
- [`contiguous_mm.py_docs.md`](./contiguous_mm.py_docs.md)
- [`triton.py_docs.md`](./triton.py_docs.md)
- [`triton_addmm.py_docs.md`](./triton_addmm.py_docs.md)


## Cross-References

- **File Documentation**: `aten.py_docs.md`
- **Keyword Index**: `aten.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/template_heuristics`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/template_heuristics`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_inductor/template_heuristics`):

- [`decompose_k.py_docs.md_docs.md`](./decompose_k.py_docs.md_docs.md)
- [`registry.py_kw.md_docs.md`](./registry.py_kw.md_docs.md)
- [`params.py_docs.md_docs.md`](./params.py_docs.md_docs.md)
- [`aten.py_kw.md_docs.md`](./aten.py_kw.md_docs.md)
- [`decompose_k.py_kw.md_docs.md`](./decompose_k.py_kw.md_docs.md)
- [`base.py_kw.md_docs.md`](./base.py_kw.md_docs.md)
- [`triton.py_kw.md_docs.md`](./triton.py_kw.md_docs.md)
- [`cutedsl.py_docs.md_docs.md`](./cutedsl.py_docs.md_docs.md)
- [`gemm.py_kw.md_docs.md`](./gemm.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `aten.py_docs.md_docs.md`
- **Keyword Index**: `aten.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
