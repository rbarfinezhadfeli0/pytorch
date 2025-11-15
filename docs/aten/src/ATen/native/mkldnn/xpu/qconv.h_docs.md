# Documentation: `aten/src/ATen/native/mkldnn/xpu/qconv.h`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/xpu/qconv.h`
- **Size**: 3,579 bytes (3.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Config.h>
#include <ATen/Tensor.h>

namespace at::native::xpu {
class QConvoneDNNXPU final {
 public:
  C10_API static at::Tensor run_pointwise(
      at::Tensor act,
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double inv_output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view attr,
      torch::List<std::optional<at::Scalar>> scalars,
      std::optional<std::string_view> algorithm);

  C10_API static at::Tensor run_pointwise_tensor(
      at::Tensor act,
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view attr,
      torch::List<std::optional<at::Scalar>> scalars,
      std::optional<std::string_view> algorithm);

  C10_API static at::Tensor run_pointwise_binary(
      at::Tensor act,
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor accum,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double accum_scale,
      int64_t accum_zero_point,
      std::string_view binary_attr,
      std::optional<at::Scalar> alpha,
      std::optional<std::string_view> unary_attr,
      torch::List<std::optional<at::Scalar>> unary_scalars,
      std::optional<std::string_view> unary_algorithm);

  C10_API static at::Tensor run_pointwise_binary_tensor(
      at::Tensor act,
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor accum,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double accum_scale,
      int64_t accum_zero_point,
      std::string_view binary_attr,
      std::optional<at::Scalar> alpha,
      std::optional<std::string_view> unary_attr,
      torch::List<std::optional<at::Scalar>> unary_scalars,
      std::optional<std::string_view> unary_algorithm);

  static inline c10::ScalarType qconv_decide_out_dtype(
      const at::Tensor& act,
      const std::optional<c10::ScalarType> output_dtype);

  static at::Tensor qconv_prepack_xpu(
      at::Tensor weight,
      at::Tensor weight_scales,
      double input_scale,
      int64_t input_zero_point,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      std::optional<torch::List<int64_t>> input_shape);
};

} // namespace at::native::xpu
```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `QConvoneDNNXPU`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/Tensor.h`


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
- [`qlinear.h_docs.md`](./qlinear.h_docs.md)
- [`qconv.cpp_docs.md`](./qconv.cpp_docs.md)
- [`ScaledBlas.cpp_docs.md`](./ScaledBlas.cpp_docs.md)
- [`FusionUtils.cpp_docs.md`](./FusionUtils.cpp_docs.md)
- [`FusionUtils.h_docs.md`](./FusionUtils.h_docs.md)
- [`Conv.cpp_docs.md`](./Conv.cpp_docs.md)


## Cross-References

- **File Documentation**: `qconv.h_docs.md`
- **Keyword Index**: `qconv.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
