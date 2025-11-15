# Documentation: `docs/tools/autograd/gen_variable_factories.py_docs.md`

## File Metadata

- **Path**: `docs/tools/autograd/gen_variable_factories.py_docs.md`
- **Size**: 7,491 bytes (7.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/autograd/gen_variable_factories.py`

## File Metadata

- **Path**: `tools/autograd/gen_variable_factories.py`
- **Size**: 4,479 bytes (4.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

from __future__ import annotations

import re

import torchgen.api.python as python
from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, TensorOptionsArguments, Variant
from torchgen.utils import FileManager, mapMaybe


OPTIONAL_TYPE_PATTERN = re.compile(r"std::optional<(.+)>")
TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")


# Add 'at::' to types defined in ATen namespace, e.g. Tensor, TensorList, IntArrayRef and etc.
# TODO: maybe update the cpp argument API to take optional namespace argument?
def fully_qualified_type(argument_type: str) -> str:
    def maybe_optional_type(type: str, is_opt: bool) -> str:
        return f"std::optional<{type}>" if is_opt else type

    opt_match = OPTIONAL_TYPE_PATTERN.match(argument_type)
    is_opt = opt_match is not None
    if opt_match:
        argument_type = argument_type[opt_match.start(1) : opt_match.end(1)]
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return maybe_optional_type(argument_type, is_opt)
    index = match.start(1)
    qualified_type = f"{argument_type[:index]}at::{argument_type[index:]}"
    return maybe_optional_type(qualified_type, is_opt)


def gen_variable_factories(
    out: str, native_yaml_path: str, tags_yaml_path: str, template_path: str
) -> None:
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    factory_functions = [fn for fn in native_functions if is_factory_function(fn)]
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template(
        "variable_factories.h",
        "variable_factories.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/variable_factories.h",
            "ops_headers": [
                f"#include <ATen/ops/{fn.root_name}.h>" for fn in factory_functions
            ],
            "function_definitions": list(mapMaybe(process_function, factory_functions)),
        },
    )


@with_native_function
def is_factory_function(f: NativeFunction) -> bool:
    if Variant.function not in f.variants:
        return False

    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    return has_tensor_options or name.endswith("_like")


@with_native_function
def process_function(f: NativeFunction) -> str | None:
    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    is_factory = has_tensor_options or name.endswith("_like")

    if Variant.function not in f.variants or not is_factory:
        return None

    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False)
    sigs = [cpp_sigs.signature]
    if cpp_sigs.symint_signature is not None:
        sigs.append(cpp_sigs.symint_signature)
    r = ""
    for sig in sigs:
        formals: list[str] = []
        exprs: list[str] = []
        requires_grad = "false"
        for arg in sig.arguments():
            qualified_type = fully_qualified_type(arg.type)
            if arg.default:
                formals.append(f"{qualified_type} {arg.name} = {arg.default}")
            else:
                formals.append(f"{qualified_type} {arg.name}")

            if isinstance(arg.argument, TensorOptionsArguments):
                # note: we remove the requires_grad setting from the TensorOptions because
                # it is ignored anyways (and we actually have an assertion that it isn't set
                # which would fail otherwise). We handle requires_grad explicitly here
                # instead of passing it through to the kernel.
                exprs.append(
                    f"at::TensorOptions({arg.name}).requires_grad(::std::nullopt)"
                )
                # Manually set the requires_grad bit on the result tensor.
                requires_grad = f"{arg.name}.requires_grad()"
            else:
                exprs.append(arg.name)

        r += f"""\
inline at::Tensor {sig.name()}({", ".join(formals)}) {{
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::{sig.name()}({", ".join(exprs)}), /*requires_grad=*/{requires_grad});
}}
"""
    return r

```



## High-Level Overview


This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `fully_qualified_type`, `maybe_optional_type`, `gen_variable_factories`, `is_factory_function`, `process_function`

**Key imports**: annotations, re, torchgen.api.python as python, cpp, CppSignatureGroup, with_native_function, parse_native_yaml, NativeFunction, TensorOptionsArguments, Variant, FileManager, mapMaybe


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/autograd`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `re`
- `torchgen.api.python as python`
- `torchgen.api`: cpp
- `torchgen.api.types`: CppSignatureGroup
- `torchgen.context`: with_native_function
- `torchgen.gen`: parse_native_yaml
- `torchgen.model`: NativeFunction, TensorOptionsArguments, Variant
- `torchgen.utils`: FileManager, mapMaybe


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`tools/autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`derivatives.yaml_docs.md`](./derivatives.yaml_docs.md)
- [`gen_variable_type.py_docs.md`](./gen_variable_type.py_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`gen_autograd.py_docs.md`](./gen_autograd.py_docs.md)
- [`load_derivatives.py_docs.md`](./load_derivatives.py_docs.md)
- [`gen_view_funcs.py_docs.md`](./gen_view_funcs.py_docs.md)
- [`gen_inplace_or_view_type.py_docs.md`](./gen_inplace_or_view_type.py_docs.md)
- [`gen_python_functions.py_docs.md`](./gen_python_functions.py_docs.md)


## Cross-References

- **File Documentation**: `gen_variable_factories.py_docs.md`
- **Keyword Index**: `gen_variable_factories.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/autograd`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/tools/autograd`):

- [`gen_trace_type.py_kw.md_docs.md`](./gen_trace_type.py_kw.md_docs.md)
- [`deprecated.yaml_docs.md_docs.md`](./deprecated.yaml_docs.md_docs.md)
- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`gen_python_functions.py_kw.md_docs.md`](./gen_python_functions.py_kw.md_docs.md)
- [`deprecated.yaml_kw.md_docs.md`](./deprecated.yaml_kw.md_docs.md)
- [`context.py_kw.md_docs.md`](./context.py_kw.md_docs.md)
- [`load_derivatives.py_docs.md_docs.md`](./load_derivatives.py_docs.md_docs.md)
- [`gen_annotated_fn_args.py_kw.md_docs.md`](./gen_annotated_fn_args.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`gen_autograd_functions.py_docs.md_docs.md`](./gen_autograd_functions.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `gen_variable_factories.py_docs.md_docs.md`
- **Keyword Index**: `gen_variable_factories.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
