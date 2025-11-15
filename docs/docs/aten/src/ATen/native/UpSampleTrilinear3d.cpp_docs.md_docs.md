# Documentation: `docs/aten/src/ATen/native/UpSampleTrilinear3d.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/UpSampleTrilinear3d.cpp_docs.md`
- **Size**: 6,514 bytes (6.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/UpSampleTrilinear3d.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/UpSampleTrilinear3d.cpp`
- **Size**: 3,816 bytes (3.73 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/UpSample.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_trilinear3d.h>
#include <ATen/ops/upsample_trilinear3d_backward.h>
#include <ATen/ops/upsample_trilinear3d_backward_native.h>
#include <ATen/ops/upsample_trilinear3d_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC(upsample_trilinear3d) (
  const Tensor& input,
  IntArrayRef output_size,
  bool align_corners,
  std::optional<double> scales_d,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  auto full_output_size = native::upsample_3d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

TORCH_META_FUNC(upsample_trilinear3d_backward) (
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  bool align_corners,
  std::optional<double> scales_d,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  auto full_output_size = native::upsample_3d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 5,
      "Expected grad_output to be a tensor of dimension 5 but got: dimension ", grad_output.dim());

  for (const auto i : c10::irange(5)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output_raw_strided(0, input_size, {}, grad_output.options().memory_format(grad_output.suggest_memory_format()));
}

} // namespace at::meta
namespace at::native {

TORCH_IMPL_FUNC(upsample_trilinear3d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output
) {
  upsample_trilinear3d_kernel(kCPU, output, input, align_corners, scales_d, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input
) {
  grad_input.zero_();
  upsample_trilinear3d_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales_d, scales_h, scales_w);
}

// vec variants

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor upsample_trilinear3d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    bool align_corners,
    std::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  return at::upsample_trilinear3d(input, osize, align_corners, scale_d, scale_h, scale_w);
}

DEFINE_DISPATCH(upsample_trilinear3d_kernel);
DEFINE_DISPATCH(upsample_trilinear3d_backward_kernel);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/TensorMeta.h`
- `ATen/native/UpSample.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/upsample_trilinear3d.h`
- `ATen/ops/upsample_trilinear3d_backward.h`
- `ATen/ops/upsample_trilinear3d_backward_native.h`
- `ATen/ops/upsample_trilinear3d_native.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `UpSampleTrilinear3d.cpp_docs.md`
- **Keyword Index**: `UpSampleTrilinear3d.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `UpSampleTrilinear3d.cpp_docs.md_docs.md`
- **Keyword Index**: `UpSampleTrilinear3d.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
