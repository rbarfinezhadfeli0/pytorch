# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qconv.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qconv.h_docs.md`
- **Size**: 6,011 bytes (5.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qconv.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qconv.h`
- **Size**: 3,555 bytes (3.47 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/Tensor.h>
#include <ATen/Config.h>

namespace at {
namespace native {

class QConvoneDNN final {
 public:

  C10_API static at::Tensor run_pointwise(
      at::Tensor act, // contains quantized values but not QTensor
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
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

  C10_API static at::Tensor run_pointwise_tensor(
      at::Tensor act, // contains quantized values but not QTensor
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
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
      at::Tensor act, // contains quantized values but not QTensor
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor accum, // contains quantized values but not QTensor
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
      at::Tensor act, // contains quantized values but not QTensor
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor accum, // contains quantized values but not QTensor
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

};

} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `native`, `at`

**Classes/Structs**: `QConvoneDNN`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `ATen/Config.h`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `qconv.h_docs.md`
- **Keyword Index**: `qconv.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu`):

- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`init_qnnpack.cpp_docs.md_docs.md`](./init_qnnpack.cpp_docs.md_docs.md)
- [`qelu.cpp_kw.md_docs.md`](./qelu.cpp_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)
- [`qclamp.cpp_docs.md_docs.md`](./qclamp.cpp_docs.md_docs.md)
- [`qembeddingbag_prepack.h_docs.md_docs.md`](./qembeddingbag_prepack.h_docs.md_docs.md)
- [`qdropout.cpp_docs.md_docs.md`](./qdropout.cpp_docs.md_docs.md)
- [`qelu.cpp_docs.md_docs.md`](./qelu.cpp_docs.md_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md_docs.md`](./qembeddingbag_unpack.cpp_docs.md_docs.md)
- [`LinearUnpackImpl.cpp_kw.md_docs.md`](./LinearUnpackImpl.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `qconv.h_docs.md_docs.md`
- **Keyword Index**: `qconv.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
