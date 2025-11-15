# Documentation: `docs/aten/src/ATen/native/MaxPooling.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/MaxPooling.cpp_docs.md`
- **Size**: 5,555 bytes (5.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/MaxPooling.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/MaxPooling.cpp`
- **Size**: 2,780 bytes (2.71 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/MaxPooling.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/max_pool1d_native.h>
#include <ATen/ops/max_pool1d_with_indices.h>
#include <ATen/ops/quantized_max_pool1d.h>
#endif

namespace at::native {

DEFINE_DISPATCH(max_pool1d_stub);

namespace {

Tensor max_pool1d_impl(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  NoNamesGuard guard;

  // If stride=None then set it to kernel_size
  if (stride.empty()) {
    stride = kernel_size;
  }

  const int64_t NB = self.dim() == 3 ? self.size(-3) : 1;
  const int64_t NC = self.size(-2);
  const int64_t IW = self.size(-1);
  const int64_t KW = kernel_size[0];
  const int64_t SJ = stride[0];
  const int64_t PJ = padding[0];
  const int64_t DJ = dilation[0];

  const int64_t OW = pooling_output_shape(IW, KW, PJ, SJ, DJ, ceil_mode);
  Tensor output = at::empty({NB, NC, OW}, self.options());

  PoolingParams1D params{NB, NC, IW, OW, KW, SJ, PJ, DJ};
  max_pool1d_stub(self.device().type(), output, self, params);

  if (self.dim() == 2) {
    output.squeeze_(0);
  }

  guard.reset();
  namedinference::propagate_names(output, self);

  return output;
}

} // namespace

Tensor max_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {

  auto ndim = self.ndimension();
   TORCH_CHECK(
       (ndim == 2 && self.sym_size(0) != 0 && self.sym_size(1) != 0) ||
           (ndim == 3 && self.sym_size(1) != 0 && self.sym_size(2) != 0),
       "max_pool1d: Expected 2D or 3D (batch mode) tensor with optional 0 dim batch size for input, but got:",
       self.sym_sizes());

  if (self.is_quantized()) {
    return at::quantized_max_pool1d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }

  check_max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if ((self.requires_grad() && at::GradMode::is_enabled()) ||
      self._fw_grad(/*level */ 0).defined() ||
      !self.device().is_cpu() ||
      isTensorSubclassLike(self)) {
    // Needs indices for grad and with_indices defines CUDA dispatch
    return std::get<0>(at::max_pool1d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode));
  }
  return max_pool1d_impl(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/NamedTensorUtils.h`
- `ATen/TensorSubclassLikeUtils.h`
- `ATen/core/grad_mode.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/MaxPooling.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty.h`
- `ATen/ops/max_pool1d_native.h`
- `ATen/ops/max_pool1d_with_indices.h`
- `ATen/ops/quantized_max_pool1d.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `MaxPooling.cpp_docs.md`
- **Keyword Index**: `MaxPooling.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `MaxPooling.cpp_docs.md_docs.md`
- **Keyword Index**: `MaxPooling.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
