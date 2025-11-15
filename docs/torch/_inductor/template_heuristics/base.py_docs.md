# Documentation: `torch/_inductor/template_heuristics/base.py`

## File Metadata

- **Path**: `torch/_inductor/template_heuristics/base.py`
- **Size**: 2,497 bytes (2.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .params import DictKernelTemplateParams, KernelTemplateParams


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..kernel_inputs import KernelInputs


class TemplateConfigHeuristics:
    """Base class for generating sets of configs for an associated template."""

    def should_run(self, inputs: KernelInputs) -> bool:
        """
        hookup to check whether the configs are right to run at all e.g. you can check
        max-autotune specific to your heuristic here or other things
        If this returns False, get_template_configs will yield no configs

        Args:
            inputs: KernelInputs
        """
        return True

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[KernelTemplateParams, None, None]:
        """
        Get template configs for the given inputs.

        Prefer to override the _get_template_configs_impl method
        to leverage things like should_run
        """
        if not self.should_run(kernel_inputs):
            return

        # Generate configs and fuse with extra_kwargs
        for config_dict in self._get_template_configs_impl(kernel_inputs, op_name):
            # Fuse extra_kwargs into config
            yield DictKernelTemplateParams(config_dict)

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get template configs for the given inputs.
        This is the main entry point for template-specific logic.
        """
        # base implementation yields no entries
        yield from []

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        """
        Get extra kwargs for the given inputs/op for the template.

        Use this to return kwargs that are needed for the template, but
        do not change depending on the config/choice, but are rather
        always the same, for all configs
        """
        return {}

    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        """
        Adjust kernel inputs for the given inputs/op for the template.

        override this to adjust the kernel inputs e.g. (un)squeezing
        """
        return kernel_inputs

```



## High-Level Overview

"""Base class for generating sets of configs for an associated template."""    def should_run(self, inputs: KernelInputs) -> bool:

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TemplateConfigHeuristics`

**Functions defined**: `should_run`, `get_template_configs`, `_get_template_configs_impl`, `get_extra_kwargs`, `adjust_kernel_inputs`

**Key imports**: annotations, Any, TYPE_CHECKING, DictKernelTemplateParams, KernelTemplateParams, Generator, KernelInputs


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/template_heuristics`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any, TYPE_CHECKING
- `.params`: DictKernelTemplateParams, KernelTemplateParams
- `collections.abc`: Generator
- `..kernel_inputs`: KernelInputs


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

Files in the same folder (`torch/_inductor/template_heuristics`):

- [`aten.py_docs.md`](./aten.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`params.py_docs.md`](./params.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`cutedsl.py_docs.md`](./cutedsl.py_docs.md)
- [`decompose_k.py_docs.md`](./decompose_k.py_docs.md)
- [`contiguous_mm.py_docs.md`](./contiguous_mm.py_docs.md)
- [`triton.py_docs.md`](./triton.py_docs.md)
- [`triton_addmm.py_docs.md`](./triton_addmm.py_docs.md)


## Cross-References

- **File Documentation**: `base.py_docs.md`
- **Keyword Index**: `base.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
