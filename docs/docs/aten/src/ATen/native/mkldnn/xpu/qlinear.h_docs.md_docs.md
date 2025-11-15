# Documentation: `docs/aten/src/ATen/native/mkldnn/xpu/qlinear.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/xpu/qlinear.h_docs.md`
- **Size**: 5,129 bytes (5.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/xpu/qlinear.h`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/xpu/qlinear.h`
- **Size**: 2,810 bytes (2.74 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Config.h>
#include <ATen/Tensor.h>
#include <ATen/core/List.h>

namespace at::native::xpu {

class QLinearOnednnXPU final {
 public:
  C10_API static Tensor q_linear_pointwise(
      Tensor act,
      double act_scale,
      int64_t act_zero_point,
      Tensor weight,
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view post_op_name,
      torch::List<std::optional<at::Scalar>> post_op_args,
      std::string_view post_op_algorithm);

  C10_API static Tensor q_linear_pointwise_tensor(
      Tensor act,
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor weight,
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view post_op_name,
      torch::List<std::optional<at::Scalar>> post_op_args,
      std::string_view post_op_algorithm);

  C10_API static Tensor q_linear_pointwise_binary(
      Tensor act,
      double act_scale,
      int64_t act_zero_point,
      Tensor weight,
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<at::Tensor> other,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double other_scale,
      int64_t other_zero_point,
      std::string_view binary_post_op,
      double binary_alpha,
      std::string_view unary_post_op,
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      std::string_view unary_post_op_algorithm);

  C10_API static Tensor q_linear_pointwise_binary_tensor(
      Tensor act,
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor weight,
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<at::Tensor> other,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double other_scale,
      int64_t other_zero_point,
      std::string_view binary_post_op,
      double binary_alpha,
      std::string_view unary_post_op,
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      std::string_view unary_post_op_algorithm);

  C10_API static Tensor q_linear_prepack_onednn(
      at::Tensor weight,
      std::optional<torch::List<int64_t>> input_shape);

  static inline c10::ScalarType qlinear_decide_out_dtype(
      const at::Tensor& act,
      const std::optional<c10::ScalarType> output_dtype);

}; // class QLinearOnednnXPU

} // namespace at::native::xpu

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `QLinearOnednnXPU`, `QLinearOnednnXPU`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/Tensor.h`
- `ATen/core/List.h`


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

Files in the same folder (`aten/src/ATen/native/mkldnn/xpu`):

- [`Attention.cpp_docs.md`](./Attention.cpp_docs.md)
- [`Conv.h_docs.md`](./Conv.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`qconv.cpp_docs.md`](./qconv.cpp_docs.md)
- [`ScaledBlas.cpp_docs.md`](./ScaledBlas.cpp_docs.md)
- [`qconv.h_docs.md`](./qconv.h_docs.md)
- [`FusionUtils.cpp_docs.md`](./FusionUtils.cpp_docs.md)
- [`FusionUtils.h_docs.md`](./FusionUtils.h_docs.md)
- [`Conv.cpp_docs.md`](./Conv.cpp_docs.md)


## Cross-References

- **File Documentation**: `qlinear.h_docs.md`
- **Keyword Index**: `qlinear.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/mkldnn/xpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/mkldnn/xpu`):

- [`FusionUtils.cpp_kw.md_docs.md`](./FusionUtils.cpp_kw.md_docs.md)
- [`qconv.cpp_docs.md_docs.md`](./qconv.cpp_docs.md_docs.md)
- [`Conv.h_kw.md_docs.md`](./Conv.h_kw.md_docs.md)
- [`qconv.h_docs.md_docs.md`](./qconv.h_docs.md_docs.md)
- [`ScaledBlas.cpp_docs.md_docs.md`](./ScaledBlas.cpp_docs.md_docs.md)
- [`Attention.cpp_kw.md_docs.md`](./Attention.cpp_kw.md_docs.md)
- [`FusionUtils.h_docs.md_docs.md`](./FusionUtils.h_docs.md_docs.md)
- [`Blas.cpp_docs.md_docs.md`](./Blas.cpp_docs.md_docs.md)
- [`Conv.cpp_docs.md_docs.md`](./Conv.cpp_docs.md_docs.md)
- [`qconv.cpp_kw.md_docs.md`](./qconv.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `qlinear.h_docs.md_docs.md`
- **Keyword Index**: `qlinear.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
