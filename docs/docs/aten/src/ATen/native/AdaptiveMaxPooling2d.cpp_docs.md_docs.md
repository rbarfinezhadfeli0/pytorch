# Documentation: `docs/aten/src/ATen/native/AdaptiveMaxPooling2d.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/AdaptiveMaxPooling2d.cpp_docs.md`
- **Size**: 6,160 bytes (6.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/AdaptiveMaxPooling2d.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/AdaptiveMaxPooling2d.cpp`
- **Size**: 3,556 bytes (3.47 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_max_pool2d_backward_native.h>
#include <ATen/ops/adaptive_max_pool2d_native.h>
#endif

namespace at::meta {
TORCH_META_FUNC(adaptive_max_pool2d) (const Tensor& input, IntArrayRef output_size) {
  int ndim = input.ndimension();
  TORCH_CHECK(ndim == 3 || ndim == 4,
              "adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: ",
              input.sizes());
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(input.size(i) > 0,
        "adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i,
        " being empty");
  }

  TORCH_CHECK(output_size.size() == 2,
      "adaptive_max_pool2d(): internal error: output_size.size() must be 2");

  int dimH = 1;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  if (input.ndimension() == 4) {
    sizeB = input.size(0);
    dimH++;
  }

  sizeD = input.size(dimH - 1);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  /* resize output */
  if (input.ndimension() == 3) {
    set_output_raw_strided(0, {sizeD, osizeH, osizeW}, {}, input.options());
    /* indices will contain i,j locations for each output point */
    set_output_raw_strided(1, {sizeD, osizeH, osizeW}, {}, input.options().dtype(kLong));
  } else {
    set_output_raw_strided(0, {sizeB, sizeD, osizeH, osizeW}, {}, input.options().memory_format(input.suggest_memory_format()));
    /* indices will contain i,j locations for each output point */
    set_output_raw_strided(1, {sizeB, sizeD, osizeH, osizeW}, {}, input.options().memory_format(input.suggest_memory_format()).dtype(kLong));
  }
}

TORCH_META_FUNC(adaptive_max_pool2d_backward)
(const Tensor& grad_output, const Tensor& input, const Tensor& indices) {
  int64_t ndim = grad_output.ndimension();
  TORCH_CHECK(ndim == 3 || ndim == 4,
    "adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: ", grad_output.sizes());

  at::native::adaptive_pool_empty_output_check(grad_output, "adaptive_max_pool2d_backward");

  TORCH_CHECK(input.ndimension() == indices.ndimension(),
    "expected dimensions ", input.ndimension(), " for `indices` but got dimensions ", indices.ndimension());
  TORCH_CHECK(input.dtype() == grad_output.dtype(),
    "expected dtype ", input.dtype(), " for `grad_output` but got dtype ", grad_output.dtype());
  TORCH_CHECK(indices.sizes() == grad_output.sizes(),
    "expected sizes ", indices.sizes(), " for `grad_output` but got sizes ", grad_output.sizes());

  set_output_raw_strided(0, input.sizes(), {}, input.options().memory_format(input.suggest_memory_format()));
}
} // namespace at::meta

namespace at::native {

TORCH_IMPL_FUNC(adaptive_max_pool2d_out_cpu)
(const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
  adaptive_max_pool2d_kernel(kCPU, output, indices, input, output_size);
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_cpu)
(const Tensor& grad_output, const Tensor& input, const Tensor& indices, const Tensor& grad_input) {
  grad_input.zero_();
  adaptive_max_pool2d_backward_kernel(kCPU, grad_input, grad_output, indices);
 }

DEFINE_DISPATCH(adaptive_max_pool2d_kernel);
DEFINE_DISPATCH(adaptive_max_pool2d_backward_kernel);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

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
- `ATen/native/AdaptivePooling.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/adaptive_max_pool2d_backward_native.h`
- `ATen/ops/adaptive_max_pool2d_native.h`


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

- **File Documentation**: `AdaptiveMaxPooling2d.cpp_docs.md`
- **Keyword Index**: `AdaptiveMaxPooling2d.cpp_kw.md`
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

- **File Documentation**: `AdaptiveMaxPooling2d.cpp_docs.md_docs.md`
- **Keyword Index**: `AdaptiveMaxPooling2d.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
