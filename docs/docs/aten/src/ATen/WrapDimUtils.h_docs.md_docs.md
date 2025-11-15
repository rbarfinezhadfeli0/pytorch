# Documentation: `docs/aten/src/ATen/WrapDimUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/WrapDimUtils.h_docs.md`
- **Size**: 7,328 bytes (7.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/WrapDimUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/WrapDimUtils.h`
- **Size**: 4,855 bytes (4.74 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>

namespace at {

// if dim_post_expr is 0 and wrap_scalar is true, then dim must be in the
// range [-1, 0]. This is a special case for scalar tensors and manifests in
// e.g. torch.sum(scalar_tensor, 0) Otherwise, dim should be in the range
// [-dim_post_expr, dim_post_expr-1].
using c10::maybe_wrap_dim;

inline int64_t maybe_wrap_dim(int64_t dim, TensorImpl* tensor) {
  return maybe_wrap_dim(dim, tensor->dim());
}

inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors) {
  if (tensors.empty()) {
    // can't wrap empty TensorList; rely on underlying implementation to throw
    // error if necessary.
    return dim;
  }
  return maybe_wrap_dim(dim, tensors[0].dim());
}

inline int64_t maybe_wrap_dim(
    int64_t dim,
    const std::vector<std::vector<int64_t>>& tensor_sizes) {
  if (tensor_sizes.empty()) {
    // can't wrap empty list; rely on underlying implementation to throw error
    // if necessary
    return dim;
  }
  return maybe_wrap_dim(dim, static_cast<int64_t>(tensor_sizes[0].size()));
}

// Given an array of dimensions `dims` of length `ndims`, this function "Wraps"
// each dim in-place for a tensor of rank `dim_post_expr`, allowing dims to be
// specified using negative indices.
//
// Additionally, if `wrap_scalar` is true then scalar tensors with rank 0, will
// allow dimensions in the range [-1, 0]. Otherwise, an IndexError is raised for
// dimensions not in the range [-dim_post_expr, dim_post_expr).
inline void maybe_wrap_dims_n(
    int64_t* dims,
    int64_t ndims,
    int64_t dim_post_expr,
    bool wrap_scalars = true) {
  if (dim_post_expr <= 0) {
    if (wrap_scalars) {
      dim_post_expr = 1; // this will make range [-1, 0]
    } else {
      TORCH_CHECK_INDEX(
          ndims == 0,
          "Dimension specified as ",
          dims[0],
          " but tensor has no dimensions");
      return;
    }
  }
  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  for (const auto i : c10::irange(ndims)) {
    auto& dim = dims[i];
    if (dim < min || dim > max) {
      TORCH_CHECK_INDEX(
          false,
          "Dimension out of range (expected to be in range of [",
          min,
          ", ",
          max,
          "], but got ",
          dim,
          ")");
    }
    if (dim < 0)
      dim += dim_post_expr;
  }
}

// Given a contiguous container of dimensions `dims`, this function "Wraps"
// each dim in-place for a tensor of rank `dim_post_expr`, allowing dims to be
// specified using negative indices.
//
// Additionally, if `wrap_scalar` is true then scalar tensors with rank 0, will
// allow dimensions in the range [-1, 0]. Otherwise, an IndexError is raised for
// dimensions not in the range [-dim_post_expr, dim_post_expr).
template <typename Container>
inline void maybe_wrap_dims(
    Container& dims,
    int64_t dim_post_expr,
    bool wrap_scalars = true) {
  return maybe_wrap_dims_n(
      dims.data(), dims.size(), dim_post_expr, wrap_scalars);
}

// previously, size [0] tensors were the only possible empty tensors; thus, it
// wasn't possible to cat empty tensors unless all the other tensors were
// 1-dimensional, so we allowed these tensors to be "skipped" (both for wrap
// dimension behavior and dimension size checking). We maintain this behavior
// for backwards compatibility, but only for this specific size (i.e. other
// empty sizes are not skipped).
inline int64_t legacy_cat_wrap_dim(
    int64_t dim,
    const std::vector<std::vector<int64_t>>& tensor_sizes) {
  for (auto& sizes : tensor_sizes) {
    if (sizes.size() == 1 && sizes[0] == 0) {
      continue;
    }
    return maybe_wrap_dim(dim, static_cast<int64_t>(sizes.size()));
  }
  return dim;
}

inline int64_t legacy_cat_wrap_dim_symint(
    int64_t dim,
    const std::vector<std::vector<c10::SymInt>>& tensor_sizes) {
  for (auto& sizes : tensor_sizes) {
    if (sizes.size() == 1) {
      if (TORCH_GUARD_OR_FALSE(sizes[0].sym_eq(0))) {
        continue;
      }
    }
    return maybe_wrap_dim(dim, static_cast<int64_t>(sizes.size()));
  }
  return dim;
}

inline int64_t legacy_cat_wrap_dim(
    int64_t dim,
    const MaterializedITensorListRef& tensors) {
  for (const Tensor& tensor : tensors) {
    if (tensor.dim() == 1) {
      if (TORCH_GUARD_OR_FALSE(tensor.sym_sizes()[0].sym_eq(0))) {
        continue;
      }
    }
    return maybe_wrap_dim(dim, tensor.dim());
  }
  return dim;
}

// wrap negative dims in a vector
inline void wrap_all_dims(
    std::vector<int64_t>& dims_to_wrap,
    int64_t tensor_total_dims) {
  for (const auto i : c10::irange(dims_to_wrap.size())) {
    dims_to_wrap[i] = maybe_wrap_dim(dims_to_wrap[i], tensor_total_dims);
  }
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/IListRef.h`
- `ATen/core/Tensor.h`
- `c10/core/TensorImpl.h`
- `c10/core/WrapDimMinimal.h`
- `c10/util/irange.h`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `WrapDimUtils.h_docs.md`
- **Keyword Index**: `WrapDimUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen`):

- [`Dispatch.cpp_docs.md_docs.md`](./Dispatch.cpp_docs.md_docs.md)
- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`ThreadLocalState.cpp_docs.md_docs.md`](./ThreadLocalState.cpp_docs.md_docs.md)
- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`FunctionalInverses.cpp_kw.md_docs.md`](./FunctionalInverses.cpp_kw.md_docs.md)
- [`SequenceNumber.h_kw.md_docs.md`](./SequenceNumber.h_kw.md_docs.md)
- [`ThreadLocalPythonObjects.h_docs.md_docs.md`](./ThreadLocalPythonObjects.h_docs.md_docs.md)
- [`TensorNames.h_docs.md_docs.md`](./TensorNames.h_docs.md_docs.md)
- [`LegacyBatchedTensorImpl.h_docs.md_docs.md`](./LegacyBatchedTensorImpl.h_docs.md_docs.md)
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `WrapDimUtils.h_docs.md_docs.md`
- **Keyword Index**: `WrapDimUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
