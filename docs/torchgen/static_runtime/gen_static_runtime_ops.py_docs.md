# Documentation: `torchgen/static_runtime/gen_static_runtime_ops.py`

## File Metadata

- **Path**: `torchgen/static_runtime/gen_static_runtime_ops.py`
- **Size**: 7,408 bytes (7.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import argparse
import itertools
import os
from typing import TYPE_CHECKING, TypeVar, Union

from libfb.py.log import set_simple_logging  # type: ignore[import]

from torchgen import gen
from torchgen.context import native_function_manager
from torchgen.model import DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup
from torchgen.static_runtime import config, generator


if TYPE_CHECKING:
    from collections.abc import Sequence


# Given a list of `grouped_native_functions` sorted by their op names, return a list of
# lists each of which groups ops that share the base name. For example, `mean` and
# `mean.dim` are grouped together by this function.

NativeGroupT = TypeVar(
    "NativeGroupT",
    bound=Union[NativeFunctionsGroup, NativeFunctionsViewGroup],
)


def group_functions_by_op_name(
    grouped_native_functions: Sequence[NativeGroupT],
) -> Sequence[Sequence[NativeGroupT]]:
    if not grouped_native_functions:
        return []
    groups = []

    def is_supported(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> bool:
        with native_function_manager(g):
            return generator.is_supported(g)

    eligible_ops = (g for g in grouped_native_functions if is_supported(g))
    groups = [
        list(group)
        for k, group in (
            itertools.groupby(
                eligible_ops,
                key=config.func_name_base_str,
            )
        )
    ]

    return groups


def clang_format(cpp_file_path: str) -> None:
    import subprocess

    subprocess.check_call(["clang-format", "-i", cpp_file_path])


def write_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    code = "\n".join(cpp_ops)
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch {{
namespace jit {{

{code}

}} // namespace jit
}} // namespace torch
"""
    with open(file_path, "w") as f:
        f.write(generated)
    clang_format(file_path)


def write_test_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    code = "\n".join(cpp_ops)
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>

#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

{code}

"""
    with open(file_path, "w") as f:
        f.write(generated)
    clang_format(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATen source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="caffe2/aten/src/ATen",
    )
    parser.add_argument(
        "-p",
        "--generated-ops-cpp-path",
        help="path to directory to generate op dispatcher .cpp file",
        default="caffe2/torch/csrc/jit/runtime/static/generated_ops.cpp",
    )
    parser.add_argument(
        "-t",
        "--generated-ops-test-cpp-path",
        help="path to directory to generate op dispatcher .cpp file",
        default="caffe2/benchmarks/static_runtime/test_generated_ops.cc",
    )
    options = parser.parse_args()
    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")
    parsed_yaml = gen.parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    op_generator = generator.GenOpDispatcher()
    test_case_generator = generator.GenOpTestCase()

    native_functions_groups = [
        g
        for g in gen.get_grouped_native_functions(native_functions)
        if isinstance(g, NativeFunctionsGroup)
    ]

    supported_functions_groups = group_functions_by_op_name(native_functions_groups)

    out_variant_op_result = [
        op_generator.out_variant(groups, backend_indices[DispatchKey.CPU])
        for groups in supported_functions_groups
    ]
    out_variant_test_result = [
        test_case_generator.out_variant(groups) for groups in supported_functions_groups
    ]

    native_functions_view_groups = [
        g
        for g in gen.get_grouped_by_view_native_functions(native_functions)
        if isinstance(g, NativeFunctionsViewGroup)
    ]

    supported_functions_view_groups = group_functions_by_op_name(
        native_functions_view_groups
    )

    view_op_result = [
        op_generator.view(groups, backend_indices[DispatchKey.CPU])
        for groups in supported_functions_view_groups
    ]
    view_test_result = [
        test_case_generator.view(groups) for groups in supported_functions_view_groups
    ]

    op_result = out_variant_op_result + ["\n\n"] + view_op_result
    test_result = out_variant_test_result + ["\n\n"] + view_test_result

    write_cpp(op_result, options.generated_ops_cpp_path)
    write_test_cpp(test_result, options.generated_ops_test_cpp_path)

    print(
        f"\ntotal grouped native ops: {len(gen.get_grouped_native_functions(native_functions)):d}"
    )

    print(f"grouped native ops with out variant: {len(native_functions_groups):d}")
    supported_functions_num = sum(len(groups) for groups in supported_functions_groups)
    print(f"generated functions groups with out variant: {supported_functions_num:d}")

    print(f"\nview grouped native ops: {len(native_functions_view_groups):d}")
    supported_view_functions_num = sum(
        len(groups) for groups in supported_functions_view_groups
    )
    print(f"generated functions view groups: {supported_view_functions_num:d}")

    print(
        f"\noverall generated : {supported_functions_num + supported_view_functions_num:d}"
    )


if __name__ == "__main__":
    set_simple_logging(escape_newlines=False)
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `group_functions_by_op_name`, `is_supported`, `clang_format`, `write_cpp`, `write_test_cpp`, `main`

**Key imports**: annotations, argparse, itertools, os, TYPE_CHECKING, TypeVar, Union, set_simple_logging  , gen, native_function_manager, DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup, config, generator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/static_runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `itertools`
- `os`
- `typing`: TYPE_CHECKING, TypeVar, Union
- `libfb.py.log`: set_simple_logging  
- `torchgen`: gen
- `torchgen.context`: native_function_manager
- `torchgen.model`: DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup
- `torchgen.static_runtime`: config, generator
- `collections.abc`: Sequence
- `subprocess`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torchgen/static_runtime`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`generator.py_docs.md`](./generator.py_docs.md)


## Cross-References

- **File Documentation**: `gen_static_runtime_ops.py_docs.md`
- **Keyword Index**: `gen_static_runtime_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
