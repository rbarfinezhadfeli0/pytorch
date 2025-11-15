# Documentation: `docs/torch/_inductor/codegen/debug_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/debug_utils.py_docs.md`
- **Size**: 14,667 bytes (14.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/debug_utils.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/debug_utils.py`
- **Size**: 11,317 bytes (11.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
import os
from enum import Enum
from typing import Optional, TYPE_CHECKING

import torch
from torch import dtype as torch_dtype

from .. import config
from ..virtualized import V
from .multi_kernel import MultiKernel


if TYPE_CHECKING:
    from collections.abc import Callable


log = logging.getLogger(__name__)


def _print_debugging_tensor_value_info(msg, arg):
    # helper for printing debugging stats for intermediate tensor values
    # at jit inductor level codegen
    max_numel_to_print = 64
    print(msg)
    if not isinstance(arg, torch.Tensor):
        print("Value: ", arg)
        return
    numel = arg.float().numel()
    # print the debug printing stats
    if numel <= max_numel_to_print:
        print(arg)
    print("Number of elements: ", numel)
    print("Size: ", arg.float().size())
    print("Dtype: ", arg.float().mean().item())
    print("Mean: ", arg.float().mean().item())
    print("Min: ", arg.float().min().item())
    print("Max: ", arg.float().max().item())
    print("Std: ", arg.float().std().item())


# AOTI debug printing related configs
class IntermediateValueDebuggingLevel(Enum):
    # OFF: No intermediate tensor value debug info will be printed or saved.
    OFF = "0"
    # LEVEL 1: Save all intermediate tensor values to individual `.pt` files. No debug printing will be displayed.
    SAVE_ONLY = "1"
    # LEVEL 2: Print all intermediate tensor values by default to the console. No debug saving will be performed.
    PRINT_ONLY = "2"
    # LEVEL 3: Print all kernel names to the console only. No debug saving/printing for input tensor value info will be performed.
    # This mode can be helpful in cases when you just want to pinpointing what kernel is running into a CUDA IMA issue, etc.
    PRINT_KERNEL_NAMES_ONLY = "3"


class DebugPrinterManager:
    def __init__(
        self,
        debug_printer_level,
        use_array_ref: bool,
        writeline: Optional[Callable[..., None]] = None,
        args_to_print_or_save: Optional[list[str]] = None,
        kernel_name: str = "",
        kernel=None,
        arg_signatures: Optional[list[type]] = None,
        kernel_type=None,
    ):
        self.debug_printer_level = IntermediateValueDebuggingLevel(debug_printer_level)
        self.use_array_ref = use_array_ref
        if args_to_print_or_save is None:
            args_to_print_or_save = []
        self.args_to_print_or_save = args_to_print_or_save
        self.kernel_name = kernel_name
        self.arg_signatures: Optional[list[type]] = None
        self.kernel = kernel
        self.filtered_kernel_names_to_print = self._get_debug_filtered_kernel_names()
        self.kernel_type = None

    def __enter__(self):
        self._perform_debug_print_or_save_helper(
            self.args_to_print_or_save,
            self.kernel_name,
            before_launch=True,
            arg_signatures=self.arg_signatures,
        )

    def __exit__(self, args_to_print_or_save, kernel_name, arg_signatures):
        self._perform_debug_print_or_save_helper(
            args_to_print_or_save,
            kernel_name,
            before_launch=False,
            arg_signatures=arg_signatures,
        )

    def _perform_debug_print_or_save_helper(
        self,
        args_to_print_or_save,
        kernel_name,
        before_launch,
        arg_signatures: Optional[list[type]] = None,
    ):
        if self.debug_printer_level == IntermediateValueDebuggingLevel.OFF:
            return
        if self.debug_printer_level == IntermediateValueDebuggingLevel.SAVE_ONLY:
            # by default save all the tensor values before launch
            self.codegen_intermediate_tensor_value_save(
                self.args_to_print_or_save,
                self.kernel_name,
                before_launch,
                arg_signatures=self.arg_signatures,
            )
        if self.debug_printer_level == IntermediateValueDebuggingLevel.PRINT_ONLY:
            # by default print all the tensor values before launch
            self.codegen_intermediate_tensor_value_print(
                self.args_to_print_or_save,
                self.kernel_name,
                before_launch,
                arg_signatures=self.arg_signatures,
            )
        if (
            self.debug_printer_level
            == IntermediateValueDebuggingLevel.PRINT_KERNEL_NAMES_ONLY
        ):
            # Print all kernel names to the console only
            self.codegen_intermediate_tensor_value_print(
                [],
                self.kernel_name,
                before_launch,
            )

    @functools.lru_cache  # noqa: B019
    def _get_debug_filtered_kernel_names(self) -> list[str]:
        if config.aot_inductor.filtered_kernel_names is None:
            return []
        return [
            x.strip()
            for x in config.aot_inductor.filtered_kernel_names.lower().split(",")
        ]

    def set_printer_args(
        self,
        args_to_print_or_save: list[str],
        kernel_name: str,
        arg_signatures: Optional[list[type]],
        kernel,
        kernel_type=None,
    ):
        # Note: MultiKernel debug printing is not supported for now
        if isinstance(kernel, MultiKernel):
            log.info(
                "MultiKernel type is not supported in AOTI debug printer tool yet."
            )
            self.debug_printer_level = IntermediateValueDebuggingLevel.OFF

        self.kernel_type = kernel_type
        # Note: if the kernel type is an extern kernel (or cpp kernel), we do a special handling to
        # get the list of args_to_print_or_save
        # TODO: Find a more reliable way to detect kernel args types to print for extern kernel calls
        if kernel_type == "extern":
            args_to_print_or_save_extern = [
                arg for arg in args_to_print_or_save if arg.startswith(("buf", "arg"))
            ]
            self.args_to_print_or_save = args_to_print_or_save_extern
        elif kernel_type == "cpp":
            self.args_to_print_or_save = [
                (
                    f"copy_arrayref_tensor_to_tensor({arg})"
                    if self.use_array_ref
                    else arg
                )
                for arg in args_to_print_or_save
                if arg.startswith(("buf", "arg"))
            ]
        else:
            self.args_to_print_or_save = args_to_print_or_save
        self.kernel_name = kernel_name
        self.arg_signatures = arg_signatures
        self.kernel = kernel

    def codegen_model_inputs_value_print(self, input_args_to_print: list[str]) -> None:
        if self.debug_printer_level != IntermediateValueDebuggingLevel.PRINT_ONLY:
            return
        for arg in input_args_to_print:
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.prefix.writeline(
                    f'aoti_torch_print_tensor_handle({arg}, "aoti_model_inputs - {arg}");'
                )

    def codegen_intermediate_tensor_value_save(
        self,
        args_to_save,
        kernel_name,
        before_launch=True,
        arg_signatures: Optional[list[type]] = None,
    ) -> None:
        for i, arg in enumerate(args_to_save):
            if arg_signatures is not None and not isinstance(
                arg_signatures[i], torch_dtype
            ):
                # infer from the arg data type (has torch.dtype) to see if it is a tensor type
                continue
            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.writeline(
                    f'aoti_torch_save_tensor_handle({arg}, "{arg}", "{launch_prefix}", "{kernel_name}");'
                )
            else:
                cwd = os.getcwd()
                saved_dir = cwd + "/tmp/jit_inductor/"
                if not os.path.exists(saved_dir):
                    log.info(
                        "Creating directory to save inductor intermediate tensor values."
                    )
                    os.makedirs(saved_dir)
                # Save the model to the directory
                saved_path = saved_dir + f"{launch_prefix}_{kernel_name}_{arg}.pt"
                log.info(
                    "Saved intermediate tensor %s for %s to %s",
                    arg,
                    kernel_name,
                    saved_path,
                )
                line = f"torch.save({arg}, '{saved_path}')"
                V.graph.wrapper_code.writeline(line)

    def codegen_intermediate_tensor_value_print(
        self,
        args_to_print,
        kernel_name,
        before_launch=True,
        arg_signatures: Optional[list[type]] = None,
    ) -> None:
        launch_prefix = "before_launch" if before_launch else "after_launch"

        # if the debug printing level is PRINT_KERNEL_NAMES_ONLY
        # we only print the kernel name to the console
        if (
            self.debug_printer_level
            == IntermediateValueDebuggingLevel.PRINT_KERNEL_NAMES_ONLY
        ):
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.writeline(
                    f'printf("[ {launch_prefix}: {kernel_name} ]\\n");'
                )
            return

        if self.debug_printer_level != IntermediateValueDebuggingLevel.PRINT_ONLY:
            return
        for i, arg in enumerate(args_to_print):
            # when debug printing is enabled i.e. IntermediateValueDebuggingLevel.PRINT_ONLY,
            # check if filtered kernel name list is provided
            if (
                len(self.filtered_kernel_names_to_print) > 0
                and kernel_name.lower() not in self.filtered_kernel_names_to_print
            ):
                continue
            if V.graph.cpp_wrapper:
                if arg_signatures is not None and isinstance(
                    arg_signatures[i], torch_dtype
                ):
                    # infer from the arg data type (has torch.dtype) to see if it is a tensor type
                    V.graph.wrapper_code.writeline(
                        f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                    )
                elif arg_signatures is not None and isinstance(
                    arg_signatures[i],
                    (
                        type(torch._inductor.codegen.wrapper.SymbolicCallArg),
                        type(int),
                        type(float),
                        type(bool),
                    ),
                ):
                    V.graph.wrapper_code.writeline(
                        f'printf("[  {launch_prefix} - {kernel_name} - {arg}: %ld  ]", {arg}); printf("\\\\n");'
                    )
                else:
                    if arg_signatures is None and self.kernel_type in ("cpp", "extern"):
                        V.graph.wrapper_code.writeline(
                            f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                        )
            else:
                V.graph.wrapper_code.writeline(
                    f'_print_debugging_tensor_value_info("inductor: {launch_prefix} - {kernel_name} - {arg}", {arg})'
                )

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `IntermediateValueDebuggingLevel`, `DebugPrinterManager`

**Functions defined**: `_print_debugging_tensor_value_info`, `__init__`, `__enter__`, `__exit__`, `_perform_debug_print_or_save_helper`, `_get_debug_filtered_kernel_names`, `set_printer_args`, `codegen_model_inputs_value_print`, `codegen_intermediate_tensor_value_save`, `codegen_intermediate_tensor_value_print`

**Key imports**: annotations, functools, logging, os, Enum, Optional, TYPE_CHECKING, torch, dtype as torch_dtype, config, V


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `functools`
- `logging`
- `os`
- `enum`: Enum
- `typing`: Optional, TYPE_CHECKING
- `torch`
- `..`: config
- `..virtualized`: V
- `.multi_kernel`: MultiKernel
- `collections.abc`: Callable


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


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

- **File Documentation**: `debug_utils.py_docs.md`
- **Keyword Index**: `debug_utils.py_kw.md`
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
- **Context Manager**: Implements context manager protocol


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `debug_utils.py_docs.md_docs.md`
- **Keyword Index**: `debug_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
