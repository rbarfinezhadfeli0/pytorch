# Documentation: `aten/src/ATen/native/SortingUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/SortingUtils.h`
- **Size**: 2,672 bytes (2.61 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/NumericUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

// ensure we get good values and indices for kthvalue, mode
// this will always be with the reducing dim as 1-d
inline void _reduction_with_indices_allocate_or_resize_output(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto result_sizes = self.sizes().vec();
  if (!result_sizes.empty()) {
    result_sizes[dim] = 1;
  }
  if (values.defined()) {
    TORCH_CHECK(
        self.options().type_equal(values.options()),
        "output values must be of same type as input");
    if (!keepdim && values.dim() == self.dim() - 1) {
      // unsqueeze to preserve passed in noncontiguous tensor in resize
      values.unsqueeze_(dim);
    }
    resize_output(values, result_sizes);
  } else {
    values = at::empty(result_sizes, self.options());
  }
  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    if (!keepdim && indices.dim() == self.dim() - 1) {
      // unsqueeze to preserve passed in noncontiguous tensor in resize
      indices.unsqueeze_(dim);
    }
    resize_output(indices, result_sizes);
  } else {
    indices = at::empty(result_sizes, self.options().dtype(kLong));
  }
}

// ensure we get good values and indices for topk
inline void _allocate_or_resize_output_with_indices(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    int64_t k) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto result_sizes = self.sizes().vec();
  if (!result_sizes.empty()) {
    result_sizes[dim] = k;
  }
  if (values.defined()) {
    TORCH_CHECK(
        self.options().type_equal(values.options()),
        "output values must be of same type as input");
    values.resize_(result_sizes);
  } else {
    values = at::empty(result_sizes, self.options());
  }
  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    indices.resize_(result_sizes);
  } else {
    indices = at::empty(result_sizes, self.options().dtype(kLong));
  }
}

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

- `ATen/NumericUtils.h`
- `ATen/native/Resize.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/ops/empty.h`


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

- **File Documentation**: `SortingUtils.h_docs.md`
- **Keyword Index**: `SortingUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
