# Documentation: `torch/_inductor/codegen/rocm/rocm_template.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/rocm/rocm_template.py`
- **Size**: 6,637 bytes (6.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch

from ...autotune_process import TensorMeta
from ...ir import Buffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .rocm_benchmark_request import ROCmBenchmarkRequest
from .rocm_kernel import ROCmTemplateCaller, ROCmTemplateKernel
from .rocm_template_buffer import ROCmTemplateBuffer
from .rocm_utils import DTYPE_TO_ROCM_TYPE


log = logging.getLogger(__name__)


# FIXME: unify with the CUDA version
@dataclass(frozen=True)
class ArgInfo:
    name: str
    ty: str


class ROCmTemplate(KernelTemplate):
    index_counter = itertools.count()
    gfx9_threads_per_warp = 64

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        input_reorder: Optional[list[int]] = None,
    ) -> None:
        """

        Baseclass for ROCm C++ Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the ROCmTemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(  # type: ignore[override]
        self,
        **kwargs,
    ) -> ROCmTemplateCaller:
        """
        Generates the ROCm template caller object for the given GEMM template and operation. This ROCmTemplateCaller
        may be used to call and benchmark the generated ROCm kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A ROCmTemplateCaller object representing the generated ROCm template caller.
        """
        kernel_name = f"rocm_{self.name}"
        kernel_hash_name = f"rocm_{self.name}_{next(self.index_counter)}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)),
            ROCmTemplateKernel(
                kernel_name=kernel_name,
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            ) as kernel,
        ):
            code = self.render(kernel=kernel, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Autotune key: %s, Generated Code:\n%s", kernel_hash_name, code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(DTYPE_TO_ROCM_TYPE),
                kernel.args.python_argdefs(),
            )

        input_reorder = (
            self.input_reorder
            if self.input_reorder is not None
            else list(range(len(self.input_nodes)))
        )
        expected_args = list(
            unique(self.input_nodes[idx].get_name() for idx in input_reorder)
        )
        expected_args.extend([self.output_node.get_name()])
        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )

        size_args = (
            self.size_args() if hasattr(self, "size_args") else ()
        )  # subclass should define def size_args()
        size_args_ints = [
            V.graph.sizevars.size_hint(arg) for arg in size_args
        ]  # resolve to ints for benchmarking
        # The runtime args come right after the size args
        runtime_args = self.get_runtime_arg_values(**kwargs)
        extra_args = size_args_ints + runtime_args
        bmreq = ROCmBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ROCmTemplateBuffer,
            epilogue_nodes: Optional[Sequence[IRNode]] = None,
        ):
            kernel = ROCmTemplateKernel(
                kernel_name="KERNEL_NAME",
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            )
            return kernel, render

        return ROCmTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
            kwargs,
        )

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                #include <exception>
                #include <iostream>
                #include <memory>
                #include <random>
                #include <vector>
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                // We compile all models with -fvisibility=hidden. Any symbols that need to be
                // exposed in the final shared library must be declared with PT_EXPORT to make
                // them visible.
                #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
                #define PT_EXPORT __attribute__((__visibility__("default")))
                #else
                #ifdef _WIN32
                #define PT_EXPORT __declspec(dllexport)
                #else
                #define PT_EXPORT
                #endif
                #endif
            """
        )
        return res

    def render(self, **kwargs) -> str:
        raise NotImplementedError

    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return []

    def get_runtime_arg_values(self, **kwargs) -> list[Any]:
        return []

```



## High-Level Overview

"""        Baseclass for ROCm C++ Templates, derived from KernelTemplate. Not to be instantiated directly.        Args:            name (str): The name of the ROCmTemplate object.            input_nodes (List[IRNode]): A list of input IRNodes.            layout (Layout): The layout of the output buffer / tensor.            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

This Python file contains 5 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ArgInfo`, `ROCmTemplate`

**Functions defined**: `__init__`, `generate`, `size_args`, `make_kernel_render`, `header`, `globals`, `render`, `get_runtime_arg_info`, `get_runtime_arg_values`

**Key imports**: functools, itertools, logging, Sequence, dataclass, Any, Optional, patch, TensorMeta, Buffer, IRNode, Layout, IndentedBuffer, unique


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen/rocm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `itertools`
- `logging`
- `collections.abc`: Sequence
- `dataclasses`: dataclass
- `typing`: Any, Optional
- `unittest.mock`: patch
- `...autotune_process`: TensorMeta
- `...ir`: Buffer, IRNode, Layout
- `...utils`: IndentedBuffer, unique
- `...virtualized`: V
- `..common`: KernelTemplate
- `.rocm_benchmark_request`: ROCmBenchmarkRequest
- `.rocm_kernel`: ROCmTemplateCaller, ROCmTemplateKernel
- `.rocm_template_buffer`: ROCmTemplateBuffer
- `.rocm_utils`: DTYPE_TO_ROCM_TYPE


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/_inductor/codegen/rocm`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ck_tile_universal_gemm_template.py_docs.md`](./ck_tile_universal_gemm_template.py_docs.md)
- [`rocm_benchmark_request.py_docs.md`](./rocm_benchmark_request.py_docs.md)
- [`rocm_template_buffer.py_docs.md`](./rocm_template_buffer.py_docs.md)
- [`ck_conv_template.py_docs.md`](./ck_conv_template.py_docs.md)
- [`ck_tile_template.py_docs.md`](./ck_tile_template.py_docs.md)
- [`rocm_cpp_scheduling.py_docs.md`](./rocm_cpp_scheduling.py_docs.md)
- [`ck_template.py_docs.md`](./ck_template.py_docs.md)
- [`rocm_utils.py_docs.md`](./rocm_utils.py_docs.md)


## Cross-References

- **File Documentation**: `rocm_template.py_docs.md`
- **Keyword Index**: `rocm_template.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
