# Documentation: `docs/aten/src/ATen/native/UpSampleNearest1d.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/UpSampleNearest1d.cpp_docs.md`
- **Size**: 7,809 bytes (7.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/UpSampleNearest1d.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/UpSampleNearest1d.cpp`
- **Size**: 4,930 bytes (4.81 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/UpSample.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact1d.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/upsample_nearest1d.h>
#include <ATen/ops/upsample_nearest1d_backward.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/upsample_nearest1d_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC(upsample_nearest1d) (
    const Tensor& input, IntArrayRef output_size, std::optional<double> scales
) {
  auto full_output_size = native::upsample_1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options());
}

TORCH_META_FUNC(_upsample_nearest_exact1d) (
  const Tensor& input, IntArrayRef output_size, std::optional<double> scales
) {
  auto full_output_size = native::upsample_1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options());
}

TORCH_META_FUNC(upsample_nearest1d_backward) (
    const Tensor& grad_output, IntArrayRef output_size, IntArrayRef input_size, std::optional<double> scales
) {
  auto full_output_size = native::upsample_1d_common_check(input_size, output_size);

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

TORCH_META_FUNC(_upsample_nearest_exact1d_backward) (
  const Tensor& grad_output, IntArrayRef output_size, IntArrayRef input_size, std::optional<double> scales
) {
  auto full_output_size = native::upsample_1d_common_check(input_size, output_size);

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

} // namespace at::meta


namespace at::native {

TORCH_IMPL_FUNC(upsample_nearest1d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    std::optional<double> scales,
    const Tensor& output
) {
  upsample_nearest1d_kernel(kCPU, output, input, scales);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    std::optional<double> scales,
    const Tensor& output
) {
  _upsample_nearest_exact1d_kernel(kCPU, output, input, scales);
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    const Tensor& grad_input
) {
  grad_input.zero_();
  upsample_nearest1d_backward_kernel(kCPU, grad_input, grad_output, scales);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    const Tensor& grad_input
) {
  grad_input.zero_();
  _upsample_nearest_exact1d_backward_kernel(kCPU, grad_input, grad_output, scales);
}

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

// vec variants

Tensor upsample_nearest1d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    std::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_w = get_scale_value(scale_factors, 0);
  return at::upsample_nearest1d(input, osize, scale_w);
}

Tensor _upsample_nearest_exact1d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    std::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_w = get_scale_value(scale_factors, 0);
  return at::_upsample_nearest_exact1d(input, osize, scale_w);
}

DEFINE_DISPATCH(upsample_nearest1d_kernel);
DEFINE_DISPATCH(_upsample_nearest_exact1d_kernel);
DEFINE_DISPATCH(upsample_nearest1d_backward_kernel);
DEFINE_DISPATCH(_upsample_nearest_exact1d_backward_kernel);

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

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/TensorMeta.h`
- `ATen/TensorUtils.h`
- `ATen/native/UpSample.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_upsample_nearest_exact1d.h`
- `ATen/ops/_upsample_nearest_exact1d_backward.h`
- `ATen/ops/_upsample_nearest_exact1d_backward_native.h`
- `ATen/ops/_upsample_nearest_exact1d_native.h`
- `ATen/ops/upsample_nearest1d.h`
- `ATen/ops/upsample_nearest1d_backward.h`
- `ATen/ops/upsample_nearest1d_backward_native.h`
- `ATen/ops/upsample_nearest1d_native.h`


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

- **File Documentation**: `UpSampleNearest1d.cpp_docs.md`
- **Keyword Index**: `UpSampleNearest1d.cpp_kw.md`
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

- **File Documentation**: `UpSampleNearest1d.cpp_docs.md_docs.md`
- **Keyword Index**: `UpSampleNearest1d.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
