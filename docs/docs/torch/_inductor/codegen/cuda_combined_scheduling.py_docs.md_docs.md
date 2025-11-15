# Documentation: `docs/torch/_inductor/codegen/cuda_combined_scheduling.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/cuda_combined_scheduling.py_docs.md`
- **Size**: 10,102 bytes (9.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/cuda_combined_scheduling.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/cuda_combined_scheduling.py`
- **Size**: 6,256 bytes (6.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .cutedsl.cutedsl_scheduling import CuteDSLScheduling
from .rocm.rocm_cpp_scheduling import ROCmCPPScheduling
from .triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from sympy import Expr

    import torch
    from torch.utils._ordered_set import OrderedSet

    from .common import BackendFeature

    _IntLike: TypeAlias = Union[int, Expr]


class CUDACombinedScheduling(BaseScheduling):
    """
    Scheduler for CUDA Kernels, which delegates calls as appropriate
    to the CUDA-C++ and Triton Schedulers, which both work for CUDA devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CUDA C++ code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Optional[Scheduler]) -> None:
        super().__init__(scheduler)
        self._triton_scheduling = TritonScheduling(scheduler)
        self._cuda_cpp_scheduling = CUDACPPScheduling(scheduler)
        self._rocm_cpp_scheduling = ROCmCPPScheduling(scheduler)
        self._cutedsl_scheduling = CuteDSLScheduling(scheduler)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return self._triton_scheduling.get_backend_features(device)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        if self._cuda_cpp_scheduling.is_cuda_cpp_template(node):
            return self._cuda_cpp_scheduling
        if self._rocm_cpp_scheduling.is_rocm_cpp_template(node):
            return self._rocm_cpp_scheduling
        if self._cutedsl_scheduling.is_cutedsl_template(node):
            return self._cutedsl_scheduling
        return self._triton_scheduling

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self._cuda_cpp_scheduling.can_fuse_vertical(node1, node2):
            return True
        elif self._cuda_cpp_scheduling.is_cuda_cpp_template(
            node1
        ) or self._cuda_cpp_scheduling.is_cuda_cpp_template(node2):
            return False
        # CuteDSL doesn't support vertical fusion currently
        elif self._cutedsl_scheduling.is_cutedsl_template(
            node1
        ) or self._cutedsl_scheduling.is_cutedsl_template(node2):
            return False
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        for node in (node1, node2):
            if self._cuda_cpp_scheduling.is_cuda_cpp_template(node):
                return self._cuda_cpp_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
            if self._cutedsl_scheduling.is_cutedsl_template(node):
                return self._cutedsl_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

    def group_fn(
        self, sizes: Sequence[Sequence[_IntLike]]
    ) -> tuple[tuple[_IntLike, ...], ...]:
        return self._triton_scheduling.group_fn(sizes)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> Optional[str]:
        if self._cuda_cpp_scheduling.is_cuda_cpp_template(template_node):
            assert not prologue_nodes
            return self._cuda_cpp_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._rocm_cpp_scheduling.is_rocm_cpp_template(template_node):
            assert not epilogue_nodes
            assert not prologue_nodes
            return self._rocm_cpp_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._cutedsl_scheduling.is_cutedsl_template(template_node):
            # TODO remove this when we add epilogue support
            assert not epilogue_nodes
            assert not prologue_nodes
            return self._cutedsl_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        else:
            return self._triton_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )

    def codegen_mix_order_reduction(self, node):
        return self._triton_scheduling.codegen_mix_order_reduction(node)

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None:
        return self._triton_scheduling.codegen_node(node)

    def codegen_sync(self) -> None:
        return self._triton_scheduling.codegen_sync()

    def flush(self) -> None:
        return self._triton_scheduling.flush()

    def codegen_combo_kernel(self, *args: Any, **kwargs: Any) -> None:
        return self._triton_scheduling.codegen_combo_kernel(*args, **kwargs)

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        return self._triton_scheduling.benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        return self._triton_scheduling.benchmark_codegened_module(module)

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[Any],
        benchmark_kernel: bool = False,
        hint_override: Optional[int] = None,
    ) -> str:
        return self._triton_scheduling.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel, hint_override=hint_override
        )

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode]
    ) -> tuple[float, float, list[Optional[str]]]:
        return self._triton_scheduling.benchmark_combo_kernel(node_list)

```



## High-Level Overview

"""    Scheduler for CUDA Kernels, which delegates calls as appropriate    to the CUDA-C++ and Triton Schedulers, which both work for CUDA devices    and use a unified-wrapper for codegen.    If Scheduling code needs to be specialized for the case of mixed Triton / CUDA C++ code,    this would also be the place to do it.

This Python file contains 1 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CUDACombinedScheduling`

**Functions defined**: `__init__`, `get_backend_features`, `choose_node_backend`, `can_fuse_vertical`, `can_fuse_horizontal`, `group_fn`, `codegen_template`, `codegen_mix_order_reduction`, `codegen_node`, `codegen_sync`, `flush`, `codegen_combo_kernel`, `benchmark_fused_nodes`, `benchmark_codegened_module`, `generate_kernel_code_from_nodes`, `benchmark_combo_kernel`

**Key imports**: annotations, Any, Optional, TYPE_CHECKING, Union, CUDACPPScheduling, CuteDSLScheduling, ROCmCPPScheduling, TritonScheduling, Sequence, TypeAlias, Expr, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any, Optional, TYPE_CHECKING, Union
- `.cuda.cuda_cpp_scheduling`: CUDACPPScheduling
- `.cutedsl.cutedsl_scheduling`: CuteDSLScheduling
- `.rocm.rocm_cpp_scheduling`: ROCmCPPScheduling
- `.triton`: TritonScheduling
- `collections.abc`: Sequence
- `sympy`: Expr
- `torch`
- `torch.utils._ordered_set`: OrderedSet
- `.common`: BackendFeature


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/_inductor/codegen`):

- [`cpp_wrapper_mps.py_docs.md`](./cpp_wrapper_mps.py_docs.md)
- [`wrapper_fxir.py_docs.md`](./wrapper_fxir.py_docs.md)
- [`cpp_flex_attention_template.py_docs.md`](./cpp_flex_attention_template.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`simd_kernel_features.py_docs.md`](./simd_kernel_features.py_docs.md)
- [`block_analysis.py_docs.md`](./block_analysis.py_docs.md)
- [`cpp_wrapper_cpu_array_ref.py_docs.md`](./cpp_wrapper_cpu_array_ref.py_docs.md)
- [`cpp_bmm_template.py_docs.md`](./cpp_bmm_template.py_docs.md)
- [`python_wrapper_mtia.py_docs.md`](./python_wrapper_mtia.py_docs.md)
- [`cpp_template.py_docs.md`](./cpp_template.py_docs.md)


## Cross-References

- **File Documentation**: `cuda_combined_scheduling.py_docs.md`
- **Keyword Index**: `cuda_combined_scheduling.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/_inductor/codegen`):

- [`wrapper_fxir.py_kw.md_docs.md`](./wrapper_fxir.py_kw.md_docs.md)
- [`simd.py_docs.md_docs.md`](./simd.py_docs.md_docs.md)
- [`mps_device_op_overrides.py_docs.md_docs.md`](./mps_device_op_overrides.py_docs.md_docs.md)
- [`simd_kernel_features.py_docs.md_docs.md`](./simd_kernel_features.py_docs.md_docs.md)
- [`segmented_tree.py_docs.md_docs.md`](./segmented_tree.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`wrapper.py_kw.md_docs.md`](./wrapper.py_kw.md_docs.md)
- [`mps.py_kw.md_docs.md`](./mps.py_kw.md_docs.md)
- [`cpu_device_op_overrides.py_kw.md_docs.md`](./cpu_device_op_overrides.py_kw.md_docs.md)
- [`cpp_gemm_template.py_kw.md_docs.md`](./cpp_gemm_template.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cuda_combined_scheduling.py_docs.md_docs.md`
- **Keyword Index**: `cuda_combined_scheduling.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
