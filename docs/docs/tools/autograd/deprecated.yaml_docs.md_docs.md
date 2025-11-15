# Documentation: `docs/tools/autograd/deprecated.yaml_docs.md`

## File Metadata

- **Path**: `docs/tools/autograd/deprecated.yaml_docs.md`
- **Size**: 8,616 bytes (8.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/autograd/deprecated.yaml`

## File Metadata

- **Path**: `tools/autograd/deprecated.yaml`
- **Size**: 6,250 bytes (6.10 KB)
- **Type**: YAML Configuration
- **Extension**: `.yaml`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```yaml
# Deprecated function signatures. These are exposed in Python, but not included
# in the error message suggestions.

- name: add(Tensor self, Scalar alpha, Tensor other) -> Tensor
  aten: add(self, other, alpha)

- name: add_(Tensor(a!) self, Scalar alpha, Tensor other) -> Tensor(a!)
  aten: add_(self, other, alpha)

- name: add(Tensor self, Scalar alpha, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  aten: add_out(out, self, other, alpha)

- name: addbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2) -> Tensor
  aten: addbmm(self, batch1, batch2, beta, alpha)

- name: addbmm_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor batch1, Tensor batch2) -> Tensor(a!)
  aten: addbmm_(self, batch1, batch2, beta, alpha)

- name: addbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
  aten: addbmm_out(out, self, batch1, batch2, beta, alpha)

- name: addbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2) -> Tensor
  aten: addbmm(self, batch1, batch2, beta, 1)

- name: addbmm_(Scalar beta, Tensor(a!) self, Tensor batch1, Tensor batch2) -> Tensor(a!)
  aten: addbmm_(self, batch1, batch2, beta, 1)

- name: addbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
  aten: addbmm_out(out, self, batch1, batch2, beta, 1)

- name: addcdiv(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2) -> Tensor
  aten: addcdiv(self, tensor1, tensor2, value)

- name: addcdiv_(Tensor(a!) self, Scalar value, Tensor tensor1, Tensor tensor2) -> Tensor(a!)
  aten: addcdiv_(self, tensor1, tensor2, value)

- name: addcdiv(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor(a!) out) -> Tensor(a!)
  aten: addcdiv_out(out, self, tensor1, tensor2, value)

- name: addcmul(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2) -> Tensor
  aten: addcmul(self, tensor1, tensor2, value)

- name: addcmul_(Tensor(a!) self, Scalar value, Tensor tensor1, Tensor tensor2) -> Tensor(a!)
  aten: addcmul_(self, tensor1, tensor2, value)

- name: addcmul(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor(a!) out) -> Tensor(a!)
  aten: addcmul_out(out, self, tensor1, tensor2, value)

- name: addmm(Scalar beta, Tensor self, Scalar alpha, Tensor mat1, Tensor mat2) -> Tensor
  aten: addmm(self, mat1, mat2, beta, alpha)

- name: addmm_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor mat1, Tensor mat2) -> Tensor(a!)
  aten: addmm_(self, mat1, mat2, beta, alpha)

- name: addmm(Scalar beta, Tensor self, Scalar alpha, Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
  aten: addmm_out(out, self, mat1, mat2, beta, alpha)

- name: addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2) -> Tensor
  aten: addmm(self, mat1, mat2, beta, 1)

- name: addmm_(Scalar beta, Tensor(a!) self, Tensor mat1, Tensor mat2) -> Tensor(a!)
  aten: addmm_(self, mat1, mat2, beta, 1)

- name: addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
  aten: addmm_out(out, self, mat1, mat2, beta, 1)

- name: sspaddmm(Scalar beta, Tensor self, Scalar alpha, Tensor mat1, Tensor mat2) -> Tensor
  aten: sspaddmm(self, mat1, mat2, beta, alpha)

- name: sspaddmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2) -> Tensor
  aten: sspaddmm(self, mat1, mat2, beta, 1)

- name: addmv(Scalar beta, Tensor self, Scalar alpha, Tensor mat, Tensor vec) -> Tensor
  aten: addmv(self, mat, vec, beta, alpha)

- name: addmv_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor mat, Tensor vec) -> Tensor(a!)
  aten: addmv_(self, mat, vec, beta, alpha)

- name: addmv(Scalar beta, Tensor self, Scalar alpha, Tensor mat, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
  aten: addmv_out(out, self, mat, vec, beta, alpha)

- name: addmv(Scalar beta, Tensor self, Tensor mat, Tensor vec) -> Tensor
  aten: addmv(self, mat, vec, beta, 1)

- name: addmv_(Scalar beta, Tensor(a!) self, Tensor mat, Tensor vec) -> Tensor(a!)
  aten: addmv_(self, mat, vec, beta, 1)

- name: addmv(Scalar beta, Tensor self, Tensor mat, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
  aten: addmv_out(out, self, mat, vec, beta, 1)

- name: addr(Scalar beta, Tensor self, Scalar alpha, Tensor vec1, Tensor vec2) -> Tensor
  aten: addr(self, vec1, vec2, beta, alpha)

- name: addr_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor vec1, Tensor vec2) -> Tensor(a!)
  aten: addr_(self, vec1, vec2, beta, alpha)

- name: addr(Scalar beta, Tensor self, Scalar alpha, Tensor vec1, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
  aten: addr_out(out, self, vec1, vec2, beta, alpha)

- name: addr(Scalar beta, Tensor self, Tensor vec1, Tensor vec2) -> Tensor
  aten: addr(self, vec1, vec2, beta, 1)

- name: addr_(Scalar beta, Tensor(a!) self, Tensor vec1, Tensor vec2) -> Tensor(a!)
  aten: addr_(self, vec1, vec2, beta, 1)

- name: addr(Scalar beta, Tensor self, Tensor vec1, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
  aten: addr_out(out, self, vec1, vec2, beta, 1)

- name: baddbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2) -> Tensor
  aten: baddbmm(self, batch1, batch2, beta, alpha)

- name: baddbmm_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor batch1, Tensor batch2) -> Tensor(a!)
  aten: baddbmm_(self, batch1, batch2, beta, alpha)

- name: baddbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
  aten: baddbmm_out(out, self, batch1, batch2, beta, alpha)

- name: baddbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2) -> Tensor
  aten: baddbmm(self, batch1, batch2, beta, 1)

- name: baddbmm_(Scalar beta, Tensor(a!) self, Tensor batch1, Tensor batch2) -> Tensor(a!)
  aten: baddbmm_(self, batch1, batch2, beta, 1)

- name: baddbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
  aten: baddbmm_out(out, self, batch1, batch2, beta, 1)

- name: sub(Tensor self, Scalar alpha, Tensor other) -> Tensor
  aten: sub(self, other, alpha)

- name: sub_(Tensor(a!) self, Scalar alpha, Tensor other) -> Tensor(a!)
  aten: sub_(self, other, alpha)

- name: sub(Tensor self, Scalar alpha, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  aten: sub_out(out, self, other, alpha)

```



## High-Level Overview

This file is part of the PyTorch framework located at `tools/autograd`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/autograd`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

- **File Documentation**: `deprecated.yaml_docs.md`
- **Keyword Index**: `deprecated.yaml_kw.md`
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
- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`gen_python_functions.py_kw.md_docs.md`](./gen_python_functions.py_kw.md_docs.md)
- [`deprecated.yaml_kw.md_docs.md`](./deprecated.yaml_kw.md_docs.md)
- [`context.py_kw.md_docs.md`](./context.py_kw.md_docs.md)
- [`load_derivatives.py_docs.md_docs.md`](./load_derivatives.py_docs.md_docs.md)
- [`gen_annotated_fn_args.py_kw.md_docs.md`](./gen_annotated_fn_args.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`gen_autograd_functions.py_docs.md_docs.md`](./gen_autograd_functions.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `deprecated.yaml_docs.md_docs.md`
- **Keyword Index**: `deprecated.yaml_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
