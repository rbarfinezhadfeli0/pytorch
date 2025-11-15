# Documentation: `torch/csrc/jit/tensorexpr/codegen_external.py`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/codegen_external.py`
- **Size**: 3,867 bytes (3.78 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
# mypy: ignore-errors

import argparse

import torchgen.model as model
from torchgen.gen import FileManager, parse_native_yaml


def num_leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip())


def deindent(code: str) -> str:
    lines = code.split("\n")
    min_leading_spaces = min(map(num_leading_spaces, lines))
    lines = [line[min_leading_spaces:] for line in lines]
    return "\n".join(lines)


def gen_external(native_functions_path, tags_path, external_path):
    native_functions = parse_native_yaml(native_functions_path, tags_path)
    func_decls = []
    func_registrations = []
    for func in native_functions:
        schema = func.func
        name = schema.name.name.base
        args = schema.arguments
        # Only supports extern calls for functions with out variants
        if not schema.is_out_fn():
            continue

        # Doesn't currently support functions with more than one out parameter
        if len(args.out) > 1:
            continue

        # Doesn't currently support kwarg arguments
        if (
            len(args.pre_tensor_options_kwarg_only) > 0
            or len(args.post_tensor_options_kwarg_only) > 0
        ):
            continue
        self_arg = [args.self_arg.argument] if args.self_arg is not None else []
        args = (
            list(args.pre_self_positional) + self_arg + list(args.post_self_positional)
        )
        tensor_args = [
            arg
            for arg in args
            if isinstance(arg.type, model.BaseType)
            and arg.type.name == model.BaseTy.Tensor
        ]
        if len(tensor_args) != len(args):
            continue

        arg_names = [None] * len(args)

        tensor_decls = []
        for idx, arg in enumerate(tensor_args):
            s = f"const at::Tensor& {arg.name} = tensors[{idx + 1}];"
            tensor_decls.append(s)
            arg_names[idx] = arg.name
        nl = "\n"

        # print(tensor_decls, name, arg_names)
        func_decl = f"""\
void nnc_aten_{name}(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {{
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);
  at::Tensor& r = tensors[0];
  {nl.join(tensor_decls)}
  try {{
    at::{name}_out({", ".join(["r"] + arg_names)});
  }} catch (...) {{
  }}
}}"""
        func_registration = f"""\
const static RegisterNNCExternalFunction nnc_{name}(
    "nnc_aten_{name}",
    nnc_aten_{name});"""
        func_decls.append(func_decl)
        func_registrations.append(func_registration)
    fm = FileManager(install_dir=".", template_dir=".", dry_run=False)
    fm.write_with_template(
        "external_functions_codegen.cpp",
        external_path,
        lambda: {
            "external_registrations": func_registrations,
            "external_functions": func_decls,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate annotated_fn_args script")
    parser.add_argument(
        "--native-functions",
        "--native_functions",
        help="path to native_functions.yaml",
        default="../../../../aten/src/ATen/native/native_functions.yaml",
    )
    parser.add_argument(
        "--tags",
        help="path to tags.yaml",
        default="../../../../aten/src/ATen/native/tags.yaml",
    )
    parser.add_argument(
        "--template-path",
        "--template_path",
        help="path to external_functions_codegen_template.cpp",
        default="../../../../tools/jit/templates/external_functions_codegen_template.cpp",
    )
    args = parser.parse_args()
    gen_external(args.native_functions, args.tags, args.template_path)


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `num_leading_spaces`, `deindent`, `gen_external`, `main`

**Key imports**: argparse, torchgen.model as model, FileManager, parse_native_yaml


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `torchgen.model as model`
- `torchgen.gen`: FileManager, parse_native_yaml


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

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `codegen_external.py_docs.md`
- **Keyword Index**: `codegen_external.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
