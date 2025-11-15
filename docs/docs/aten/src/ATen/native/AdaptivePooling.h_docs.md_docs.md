# Documentation: `docs/aten/src/ATen/native/AdaptivePooling.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/AdaptivePooling.h_docs.md`
- **Size**: 4,894 bytes (4.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/AdaptivePooling.h`

## File Metadata

- **Path**: `aten/src/ATen/native/AdaptivePooling.h`
- **Size**: 2,428 bytes (2.37 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <cmath>

namespace at::native {

using adaptive_avg_pooling2d_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
using adaptive_avg_pooling2d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
DECLARE_DISPATCH(adaptive_avg_pooling2d_fn, adaptive_avg_pool2d_kernel)
DECLARE_DISPATCH(adaptive_avg_pooling2d_backward_fn, adaptive_avg_pool2d_backward_kernel)

using adaptive_max_pooling2d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input, IntArrayRef output_size);
using adaptive_max_pooling2d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
DECLARE_DISPATCH(adaptive_max_pooling2d_fn, adaptive_max_pool2d_kernel)
DECLARE_DISPATCH(adaptive_max_pooling2d_backward_fn, adaptive_max_pool2d_backward_kernel)

using adaptive_avg_pooling3d_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
using adaptive_avg_pooling3d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
DECLARE_DISPATCH(adaptive_avg_pooling3d_fn, adaptive_avg_pool3d_kernel)
DECLARE_DISPATCH(adaptive_avg_pooling3d_backward_fn, adaptive_avg_pool3d_backward_kernel)

using adaptive_max_pooling3d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input, IntArrayRef output_size);
using adaptive_max_pooling3d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
DECLARE_DISPATCH(adaptive_max_pooling3d_fn, adaptive_max_pool3d_kernel)
DECLARE_DISPATCH(adaptive_max_pooling3d_backward_fn, adaptive_max_pool3d_backward_kernel)

inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

inline void adaptive_pool_empty_output_check(const Tensor& gradOutput_, const char* arg_name) {
  int64_t ndim = gradOutput_.ndimension();
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(gradOutput_.size(i) > 0,
      arg_name, "(): Expected grad_output to have non-zero size for non-batch dimensions, "
      "but grad_output has sizes ", gradOutput_.sizes(), " with dimension ", i,
      " being empty");
  }
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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
- `ATen/native/DispatchStub.h`
- `c10/util/ArrayRef.h`
- `c10/util/irange.h`
- `cmath`


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

- **File Documentation**: `AdaptivePooling.h_docs.md`
- **Keyword Index**: `AdaptivePooling.h_kw.md`
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

- **File Documentation**: `AdaptivePooling.h_docs.md_docs.md`
- **Keyword Index**: `AdaptivePooling.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
