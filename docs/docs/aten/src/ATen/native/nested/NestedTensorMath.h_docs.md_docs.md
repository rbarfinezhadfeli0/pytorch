# Documentation: `docs/aten/src/ATen/native/nested/NestedTensorMath.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/nested/NestedTensorMath.h_docs.md`
- **Size**: 5,349 bytes (5.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/nested/NestedTensorMath.h`

## File Metadata

- **Path**: `aten/src/ATen/native/nested/NestedTensorMath.h`
- **Size**: 2,718 bytes (2.65 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/macros/Macros.h>

namespace at::native {

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size);

template <typename Func>
Tensor map_nt(const Tensor& nt, Func f) {
  auto* nt_impl = get_nested_tensor_impl(nt);
  const auto& sizes = nt_impl->get_nested_sizes();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl->get_buffer()), sizes);
}
template <typename Func>
Tensor map_nt_binary(const Tensor& nt_1, const Tensor& nt_2, Func f){
  auto* nt_impl_1 = get_nested_tensor_impl(nt_1);
  auto* nt_impl_2 = get_nested_tensor_impl(nt_2);
  const auto& sizes = nt_impl_1->get_nested_sizes();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl_1->get_buffer(), nt_impl_2->get_buffer()), sizes);
}

C10_ALWAYS_INLINE std::pair<int64_t, int64_t> _check_nested_layer_norm_inputs(
    const NestedTensorImpl& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {

  const size_t normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  // Check that the normalized_shape has the exact same sizes as the last dimensions from the NestedTensor input
  // Also, compute M and N considering the idiosyncrasies of NestedTensors
  int64_t N = 1;
  for (const auto i: c10::irange(normalized_ndim)) {
    TORCH_CHECK(
      input.opt_size(-normalized_ndim + i).has_value(),
      "normalized_shape extends into irregular dimensions for the nested tensor"
    );
    TORCH_CHECK(
      normalized_shape[i] == input.opt_size(-normalized_ndim + i),
      "The shape at dimension ",
      i,
      "of normalized_shape doesn't match the input"
    );
    N *= normalized_shape[i];
  }

  const int64_t M = input.numel() / N;

  return std::make_pair(M, N);
}

Tensor reshape_nested(const Tensor& self, IntArrayRef proposed_shape);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/nested`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ATen_fwd.h`
- `ATen/NestedTensorImpl.h`
- `c10/macros/Macros.h`


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

Files in the same folder (`aten/src/ATen/native/nested`):

- [`NestedTensorBinaryOps.cpp_docs.md`](./NestedTensorBinaryOps.cpp_docs.md)
- [`NestedTensorUtils.cpp_docs.md`](./NestedTensorUtils.cpp_docs.md)
- [`NestedTensorUnaryOps.cpp_docs.md`](./NestedTensorUnaryOps.cpp_docs.md)
- [`NestedTensorBinaryOps.h_docs.md`](./NestedTensorBinaryOps.h_docs.md)
- [`NestedTensorFactories.cpp_docs.md`](./NestedTensorFactories.cpp_docs.md)
- [`NestedTensorBackward.cpp_docs.md`](./NestedTensorBackward.cpp_docs.md)
- [`NestedTensorMatmul.cpp_docs.md`](./NestedTensorMatmul.cpp_docs.md)
- [`NestedTensorTransformerFunctions.h_docs.md`](./NestedTensorTransformerFunctions.h_docs.md)
- [`NestedTensorMath.cpp_docs.md`](./NestedTensorMath.cpp_docs.md)
- [`NestedTensorTransformerUtils.h_docs.md`](./NestedTensorTransformerUtils.h_docs.md)


## Cross-References

- **File Documentation**: `NestedTensorMath.h_docs.md`
- **Keyword Index**: `NestedTensorMath.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/nested`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/nested`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/nested`):

- [`NestedTensorTransformerUtils.h_kw.md_docs.md`](./NestedTensorTransformerUtils.h_kw.md_docs.md)
- [`NestedTensorTransformerFunctions.h_kw.md_docs.md`](./NestedTensorTransformerFunctions.h_kw.md_docs.md)
- [`NestedTensorTransformerFunctions.h_docs.md_docs.md`](./NestedTensorTransformerFunctions.h_docs.md_docs.md)
- [`NestedTensorTransformerFunctions.cpp_docs.md_docs.md`](./NestedTensorTransformerFunctions.cpp_docs.md_docs.md)
- [`NestedTensorAliases.cpp_docs.md_docs.md`](./NestedTensorAliases.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`NestedTensorUtils.h_docs.md_docs.md`](./NestedTensorUtils.h_docs.md_docs.md)
- [`NestedTensorBackward.cpp_kw.md_docs.md`](./NestedTensorBackward.cpp_kw.md_docs.md)
- [`NestedTensorBinaryOps.h_docs.md_docs.md`](./NestedTensorBinaryOps.h_docs.md_docs.md)
- [`NestedTensorBinaryOps.cpp_kw.md_docs.md`](./NestedTensorBinaryOps.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `NestedTensorMath.h_docs.md_docs.md`
- **Keyword Index**: `NestedTensorMath.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
