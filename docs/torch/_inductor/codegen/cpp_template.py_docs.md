# Documentation: cpp_template.py

## File Metadata
- **Path**: `torch/_inductor/codegen/cpp_template.py`
- **Size**: 5021 bytes
- **Lines**: 140
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-defs
import ctypes
import functools
import itertools
import logging
import sys
from collections.abc import Callable, Iterable
from typing import Optional, Union
from unittest.mock import patch

import sympy

from .. import config, ir
from ..autotune_process import CppBenchmarkRequest, TensorMeta
from ..utils import IndentedBuffer, Placeholder, unique
from ..virtualized import V
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateCaller, CppTemplateKernel


log = logging.getLogger(__name__)


class CppTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ) -> None:
        super().__init__(name)
        self.input_nodes = input_nodes
        self.index = next(self.index_counter)
        self.output_node: Union[ir.Buffer, list[ir.Buffer]] = ir.Buffer(
            name=f"buf_out{self.index}", layout=layout
        )
        self.layout = layout
        self.num_threads = num_threads
        self.epilogue_creator = epilogue_creator

    def generate(self, **kwargs):
        kernel_name = f"cpp_{self.name}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)),
            patch.object(ir.FlexibleLayout, "allow_indexing", True),
            V.graph.set_current_device(self.layout.device),
            CppTemplateKernel(
                kernel_name=kernel_name, num_threads=self.num_threads
            ) as kernel,
        ):
            code = kernel.render(self, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Generated Code:\n%s", code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )

        expected_args = list(
            unique(input_node.get_name() for input_node in self.input_nodes)
        )
        if isinstance(self.output_node, Iterable):
            expected_args.extend([node.get_name() for node in self.output_node])
        else:
            expected_args.extend([self.output_node.get_name()])
        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )
        # Cast the size hint from int to ctypes.c_ulonglong explicitly
        # since in cpp kernel, we bind it to C long
        extra_args = tuple(ctypes.c_ulonglong(x) for x in extra_args)

        kernel_hash_name = f"cpp_{self.name}_{self.index}"

        # Create the BenchmarkRequest for CPP
        bmreq = CppBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            # pyrefly: ignore [bad-argument-type]
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ir.CppTemplateBuffer,
            flag_template_buffer_has_other_users: bool,
            epilogue_nodes: Optional[list[ir.IRNode]] = None,
        ):
            kernel = CppTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME), num_threads=self.num_threads
            )
            render = functools.partial(
                kernel.render,
                self,
                template_buffer_node=template_node,
                flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
                epilogue_nodes=epilogue_nodes,
                **kwargs,
            )
            return kernel, render

        return CppTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            # pyrefly: ignore [index-error]
            self.output_node[0].get_layout()
            if isinstance(self.output_node, Iterable)
            else self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.writeline("#include <torch/csrc/inductor/cpp_prefix.h>")
        # TODO: add c10::ForcedUnroll test to test_aoti_abi_check
        res.splice("""#include <c10/util/Unroll.h>""")
        res.splice("""#include <torch/csrc/inductor/aoti_torch/c/shim.h>""")
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile:
            res.writelines(["#include <torch/csrc/inductor/aoti_runtime/utils.h>"])
        return res

    def render(self, **kwargs) -> str:
        raise NotImplementedError

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): CppTemplate

### Functions
This file defines 5 function(s): __init__, generate, make_kernel_render, header, render


## Key Components

The file contains 329 words across 140 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5021 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
