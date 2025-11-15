# Documentation: `aten/src/ATen/TensorGeometry.h`

## File Metadata

- **Path**: `aten/src/ATen/TensorGeometry.h`
- **Size**: 4,556 bytes (4.45 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/core/WrapDimMinimal.h>

namespace at {

// Return if the tensor geometry represented by `sizes` and `strides` is
// contiguous Although we cache is_contiguous in tensor now, this is till useful
// because it allows checking if a particular geometry is contiguous without
// explicitly constructing a tensor, e.g., when you want to choose a kernel
// strategy based on whether a subgeometry is contiguous.
TORCH_API bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides);

struct TORCH_API TensorGeometry {
  TensorGeometry() = default;

  explicit TensorGeometry(c10::SymIntArrayRef sizes)
      : sizes_(sizes.vec()),
        strides_(sizes.size()),
        has_symbolic_sizes_strides_(
            !c10::asIntArrayRefSlowOpt(sizes).has_value()) {
    int64_t dim = static_cast<int64_t>(sizes.size());
    c10::SymInt expected_stride = 1;
    for (int64_t i = dim - 1; i >= 0; i--) {
      strides_[i] = expected_stride;
      expected_stride *= sizes_[i];
    }
    numel_ = expected_stride;
  }

  explicit TensorGeometry(const TensorBase& t)
      : sizes_(t.sym_sizes().vec()),
        strides_(t.sym_strides().vec()),
        storage_offset_(t.sym_storage_offset()),
        numel_(t.sym_numel()),
        has_symbolic_sizes_strides_(
            t.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {}

  explicit TensorGeometry(
      std::vector<at::SymInt> sizes,
      std::vector<at::SymInt> strides,
      at::SymInt storage_offset)
      : sizes_(std::move(sizes)),
        strides_(std::move(strides)),
        storage_offset_(std::move(storage_offset)) {
    recompute();
  }

  // true if the tensor is contiguous
  bool is_contiguous() const;

  int64_t dim() const {
    return static_cast<int64_t>(sizes_.size());
  }

  int64_t size(int64_t dim) const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return sizes_.at(static_cast<size_t>(dim)).as_int_unchecked();
  }
  c10::IntArrayRef sizes() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return c10::asIntArrayRefUnchecked(sizes_);
  }
  int64_t stride(int64_t dim) const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return strides_.at(static_cast<size_t>(dim)).as_int_unchecked();
  }
  c10::IntArrayRef strides() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return c10::asIntArrayRefUnchecked(strides_);
  }
  int64_t storage_offset() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return storage_offset_.as_int_unchecked();
  }
  int64_t numel() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return numel_.as_int_unchecked();
  }

  c10::SymInt sym_size(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return sizes_.at(static_cast<size_t>(dim));
  }
  c10::SymIntArrayRef sym_sizes() const {
    return sizes_;
  }
  c10::SymInt sym_stride(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return strides_.at(static_cast<size_t>(dim));
  }
  c10::SymIntArrayRef sym_strides() const {
    return strides_;
  }
  c10::SymInt sym_storage_offset() const {
    return storage_offset_;
  }
  c10::SymInt sym_numel() const {
    return numel_;
  }

  TensorGeometry transpose(int64_t dim0, int64_t dim1) {
    TensorGeometry r = *this; // copy
    TORCH_CHECK(
        dim0 < dim(),
        "transpose: dim0=",
        dim0,
        " out of range (dim=",
        dim(),
        ")")
    TORCH_CHECK(
        dim1 < dim(),
        "transpose: dim1=",
        dim1,
        " out of range (dim=",
        dim(),
        ")")
    std::swap(r.sizes_[dim0], r.sizes_[dim1]);
    std::swap(r.strides_[dim0], r.strides_[dim1]);
    return r;
  }

  std::vector<c10::SymInt>& mutable_sizes() {
    return sizes_;
  }
  std::vector<c10::SymInt>& mutable_strides() {
    return strides_;
  }
  c10::SymInt& mutable_storage_offset() {
    return storage_offset_;
  }
  void recompute() {
    // recalculate numel after a change
    c10::SymInt numel = 1;
    for (const auto& i : sizes_) {
      numel = numel * i;
    }
    numel_ = std::move(numel);
    has_symbolic_sizes_strides_ =
        !c10::asIntArrayRefSlowOpt(sizes_).has_value();
  }

 private:
  std::vector<c10::SymInt> sizes_;
  std::vector<c10::SymInt> strides_;
  c10::SymInt storage_offset_;
  c10::SymInt numel_;
  bool has_symbolic_sizes_strides_{false};
};

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 23 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/TensorBase.h`
- `c10/core/WrapDimMinimal.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `TensorGeometry.h_docs.md`
- **Keyword Index**: `TensorGeometry.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
