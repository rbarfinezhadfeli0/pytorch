# Documentation: `torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py`
- **Size**: 5,310 bytes (5.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import hashlib
import logging
from collections.abc import Sequence
from typing import cast

from torch._inductor.utils import Placeholder
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import CuteDSLTemplateBuffer
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...select_algorithm import PartialRender
from ...utils import get_fused_kernel_name, get_kernel_metadata
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer


log = logging.getLogger(__name__)


class CuteDSLScheduling(BaseScheduling):
    """
    Scheduling implementation for CuteDSL (CUTLASS Python DSL) kernels.
    This class is intended to be used in combination with other schedulers,
    and delegated to by CUDACombinedScheduling.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_cutedsl_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is a CuteDSL template."""
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CuteDSLTemplateBuffer
        )

    def is_cutedsl_fused_template(self, node: BaseSchedulerNode) -> bool:
        """Check if a node is a fused CuteDSL template."""
        return isinstance(node, FusedSchedulerNode) and self.is_cutedsl_template(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        TODO CuteDSL doesn't support vertical fusion yet.
        This could be extended in the future for epilogue fusion.
        """
        return False

    def define_kernel(self, src_code_str: str, node_schedule) -> str:
        """Produce the kernel string
        Args:
            src_code_str: The finalized kernel code string
            node_schedule: List of nodes in the schedule

        Note:
            This is a little weird since async_compile.cutedsl() has to write the string to
            a file in order to cute compile it. Feels bad to have two...
        """
        wrapper = V.graph.wrapper_code

        # Use the string as the key for caching
        if src_code_str in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code_str]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )

            kernel_hash = hashlib.sha256(src_code_str.encode("utf-8")).hexdigest()[:8]
            if fused_name == "fused":
                kernel_name = f"cutedsl_{kernel_hash}"
            else:
                kernel_name = f"cutedsl_{fused_name}_{kernel_hash}"
            wrapper.src_to_kernel[src_code_str] = kernel_name
            src_code_str = src_code_str.replace(
                str(Placeholder.KERNEL_NAME), kernel_name
            )

            _, _, kernel_path = get_path(code_hash(src_code_str), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.cutedsl({kernel_name!r}, r'''")
            compile_wrapper.splice(src_code_str, strip=True)
            compile_wrapper.writeline("''')")

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
        Codegen a CuteDSL template. Currently doesn't support fusion.
        """
        assert self.is_cutedsl_template(template_node), (
            "Template node passed to CuteDSLScheduling.codegen_template must be a "
            "SchedulerNode that wraps a CuteDSLTemplateBuffer"
        )
        # TODO remove when supported
        assert not epilogue_nodes, "CuteDSL doesn't support epilogue fusion yet"
        assert not prologue_nodes, "CuteDSL doesn't support prologue fusion yet"

        template_node = cast(SchedulerNode, template_node)
        ctb: CuteDSLTemplateBuffer = cast(CuteDSLTemplateBuffer, template_node.node)

        kernel, render = ctb.make_kernel_render(ctb)  # type: ignore[misc]
        template_node.mark_run()
        src_code = render()
        # Finalize PartialRender if needed
        if isinstance(src_code, PartialRender):
            src_code_str = src_code.finalize_all()
        else:
            src_code_str = src_code

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(src_code_str, node_schedule)
        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()

```



## High-Level Overview

"""    Scheduling implementation for CuteDSL (CUTLASS Python DSL) kernels.    This class is intended to be used in combination with other schedulers,    and delegated to by CUDACombinedScheduling.

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CuteDSLScheduling`

**Functions defined**: `get_backend_features`, `is_cutedsl_template`, `is_cutedsl_fused_template`, `can_fuse_vertical`, `define_kernel`, `codegen_template`

**Key imports**: hashlib, logging, Sequence, cast, Placeholder, OrderedSet, config, code_hash, get_path, CuteDSLTemplateBuffer, PartialRender


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen/cutedsl`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `hashlib`
- `logging`
- `collections.abc`: Sequence
- `typing`: cast
- `torch._inductor.utils`: Placeholder
- `torch.utils._ordered_set`: OrderedSet
- `...`: config
- `...codecache`: code_hash, get_path
- `...ir`: CuteDSLTemplateBuffer
- `...select_algorithm`: PartialRender
- `...utils`: get_fused_kernel_name, get_kernel_metadata
- `...virtualized`: V
- `..common`: BackendFeature, IndentedBuffer


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

Files in the same folder (`torch/_inductor/codegen/cutedsl`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`cutedsl_kernel.py_docs.md`](./cutedsl_kernel.py_docs.md)
- [`cutedsl_template.py_docs.md`](./cutedsl_template.py_docs.md)
- [`_cutedsl_utils.py_docs.md`](./_cutedsl_utils.py_docs.md)
- [`cutedsl_op_overrides.py_docs.md`](./cutedsl_op_overrides.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)


## Cross-References

- **File Documentation**: `cutedsl_scheduling.py_docs.md`
- **Keyword Index**: `cutedsl_scheduling.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
