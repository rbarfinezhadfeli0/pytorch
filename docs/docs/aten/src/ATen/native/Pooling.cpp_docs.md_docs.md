# Documentation: `docs/aten/src/ATen/native/Pooling.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Pooling.cpp_docs.md`
- **Size**: 8,625 bytes (8.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Pooling.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Pooling.cpp`
- **Size**: 5,562 bytes (5.43 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_avg_pool1d_native.h>
#include <ATen/ops/adaptive_avg_pool2d.h>
#include <ATen/ops/adaptive_max_pool1d_native.h>
#include <ATen/ops/adaptive_max_pool2d.h>
#include <ATen/ops/avg_pool1d_native.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/max_pool1d_with_indices_native.h>
#include <ATen/ops/max_pool2d_native.h>
#include <ATen/ops/max_pool2d_with_indices.h>
#include <ATen/ops/max_pool3d_native.h>
#include <ATen/ops/max_pool3d_with_indices.h>
#include <ATen/ops/mkldnn_max_pool2d.h>
#include <ATen/ops/mkldnn_max_pool3d.h>
#include <ATen/ops/quantized_max_pool2d.h>
#include <ATen/ops/quantized_max_pool3d.h>
#endif

#include <tuple>

namespace at::native {

static void check1d(
    const char* function_name,
    const char* argument_name,
    IntArrayRef x) {
  TORCH_CHECK(
      x.size() == 1,
      function_name, "() argument '", argument_name,
      "' should contain one int (got ", x.size(), ")");
}

Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) {
  checkDimRange("adaptive_avg_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("adaptive_avg_pool1d", "output_size", output_size);

  auto output = at::adaptive_avg_pool2d(
      self.unsqueeze(-2),
      {1, output_size[0]});

  return output.squeeze(-2);
}

std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size) {
  checkDimRange("adaptive_max_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("adaptive_max_pool1d", "output_size", output_size);

  int ndim = self.ndimension();
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        self.sym_size(i) > 0,
        "adaptive_max_pool1d(): ",
        "Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        self.sym_sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  auto [output, indices] = at::adaptive_max_pool2d(
      self.unsqueeze(-2),
      {1, output_size[0]});

  return std::make_tuple(output.squeeze(-2), indices.squeeze(-2));
}

std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDimRange("max_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("max_pool1d", "kernel_size", kernel_size);
  check1d("max_pool1d", "stride", stride);
  check1d("max_pool1d", "padding", padding);
  check1d("max_pool1d", "dilation", dilation);

  NoNamesGuard guard;

  auto [output, indices] = at::max_pool2d_with_indices(
      self.unsqueeze(-2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      {1, dilation[0]},
      ceil_mode);

  output  = output.squeeze(-2);
  indices = indices.squeeze(-2);

  guard.reset();
  namedinference::propagate_names(output, self);
  namedinference::propagate_names(indices, self);

  return std::make_tuple(output, indices);
}

Tensor avg_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDimRange("avg_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("avg_pool1d", "kernel_size", kernel_size);
  check1d("avg_pool1d", "stride", stride);
  check1d("avg_pool1d", "padding", padding);

  auto output = at::avg_pool2d(
      self.unsqueeze(-2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      ceil_mode,
      count_include_pad);

  return output.squeeze(-2);
}

Tensor max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_quantized()) {
    return at::quantized_max_pool2d(self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
#if defined(C10_MOBILE)
  if(xnnpack::use_max_pool2d(self, kernel_size, padding, stride,
                             dilation, ceil_mode)) {
    return xnnpack::max_pool2d(
        self, kernel_size, padding, stride, dilation, ceil_mode);
  }
#endif
  auto output_and_indices = at::max_pool2d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::get<0>(output_and_indices);
}

Tensor max_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_quantized()) {
    return at::quantized_max_pool3d(self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool3d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  auto output_and_indices = at::max_pool3d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::get<0>(output_and_indices);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

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
- `ATen/TensorUtils.h`
- `ATen/NamedTensorUtils.h`
- `ATen/native/xnnpack/Engine.h`
- `c10/util/Exception.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/adaptive_avg_pool1d_native.h`
- `ATen/ops/adaptive_avg_pool2d.h`
- `ATen/ops/adaptive_max_pool1d_native.h`
- `ATen/ops/adaptive_max_pool2d.h`
- `ATen/ops/avg_pool1d_native.h`
- `ATen/ops/avg_pool2d.h`
- `ATen/ops/max_pool1d_with_indices_native.h`
- `ATen/ops/max_pool2d_native.h`
- `ATen/ops/max_pool2d_with_indices.h`
- `ATen/ops/max_pool3d_native.h`
- `ATen/ops/max_pool3d_with_indices.h`
- `ATen/ops/mkldnn_max_pool2d.h`
- `ATen/ops/mkldnn_max_pool3d.h`
- `ATen/ops/quantized_max_pool2d.h`
- `ATen/ops/quantized_max_pool3d.h`
- `tuple`


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

- **File Documentation**: `Pooling.cpp_docs.md`
- **Keyword Index**: `Pooling.cpp_kw.md`
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

- **File Documentation**: `Pooling.cpp_docs.md_docs.md`
- **Keyword Index**: `Pooling.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
