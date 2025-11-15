# Documentation: `torch/_inductor/codegen/rocm/rocm_cpp_scheduling.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/rocm/rocm_cpp_scheduling.py`
- **Size**: 3,878 bytes (3.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import logging
from collections.abc import Sequence
from typing import cast

from ... import config
from ...codecache import code_hash, get_path
from ...scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer
from .rocm_template_buffer import ROCmTemplateBuffer


log = logging.getLogger(__name__)


class ROCmCPPScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for ROCm C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and ROCm C++ specific template code generation.
    """

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_rocm_cpp_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ROCmTemplateBuffer
        )

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(["rocm", fused_name, wrapper.next_kernel_suffix()])
            # use the original src_code as the key
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.rocm(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline(
                f"''', 'so', aot_compile={str(V.graph.aot_mode)})"
            )

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a ROCm template, possibly with fused epilogues
        """
        assert self.is_rocm_cpp_template(template_node), (
            "Template node passed to ROCmScheduler.codegen_template must be a SchedulerNode that wraps a ROCmTemplateBuffer"
        )
        template_node = cast(SchedulerNode, template_node)
        _, (_numel, rnumel) = template_node.group
        assert rnumel == 1
        ctb: ROCmTemplateBuffer = cast(ROCmTemplateBuffer, template_node.node)
        kernel, render = ctb.make_kernel_render(ctb)  # type: ignore[misc]
        with kernel:
            template_node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(src_code, node_schedule)
        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()

```



## High-Level Overview

"""    Partial Scheduling implementation for ROCm C++ Kernels.    This class is intended to be used in combination with TritonScheduling,    and delegated to by CUDACombinedScheduling.    It handles fusion decisions and ROCm C++ specific template code generation.

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ROCmCPPScheduling`

**Functions defined**: `group_fn`, `is_rocm_cpp_template`, `can_fuse_vertical`, `define_kernel`, `codegen_template`

**Key imports**: logging, Sequence, cast, config, code_hash, get_path, BaseSchedulerNode, BaseScheduling, SchedulerNode, get_fused_kernel_name, get_kernel_metadata, sympy_product, V, IndentedBuffer, ROCmTemplateBuffer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen/rocm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `collections.abc`: Sequence
- `typing`: cast
- `...`: config
- `...codecache`: code_hash, get_path
- `...scheduler`: BaseSchedulerNode, BaseScheduling, SchedulerNode
- `...utils`: get_fused_kernel_name, get_kernel_metadata, sympy_product
- `...virtualized`: V
- `..common`: IndentedBuffer
- `.rocm_template_buffer`: ROCmTemplateBuffer


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/_inductor/codegen/rocm`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ck_tile_universal_gemm_template.py_docs.md`](./ck_tile_universal_gemm_template.py_docs.md)
- [`rocm_benchmark_request.py_docs.md`](./rocm_benchmark_request.py_docs.md)
- [`rocm_template_buffer.py_docs.md`](./rocm_template_buffer.py_docs.md)
- [`ck_conv_template.py_docs.md`](./ck_conv_template.py_docs.md)
- [`rocm_template.py_docs.md`](./rocm_template.py_docs.md)
- [`ck_tile_template.py_docs.md`](./ck_tile_template.py_docs.md)
- [`ck_template.py_docs.md`](./ck_template.py_docs.md)
- [`rocm_utils.py_docs.md`](./rocm_utils.py_docs.md)


## Cross-References

- **File Documentation**: `rocm_cpp_scheduling.py_docs.md`
- **Keyword Index**: `rocm_cpp_scheduling.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
