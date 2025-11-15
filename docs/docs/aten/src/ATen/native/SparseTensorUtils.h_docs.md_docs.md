# Documentation: `docs/aten/src/ATen/native/SparseTensorUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/SparseTensorUtils.h_docs.md`
- **Size**: 9,141 bytes (8.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/SparseTensorUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/SparseTensorUtils.h`
- **Size**: 6,500 bytes (6.35 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::sparse {

// Just for documentary purposes
using SparseTensor = Tensor;
using SparseType = Type;

// This is an internal utility function for getting at the SparseTensorImpl,
// so that we can write sparse tensor specific accessors for special fields
// in SparseTensor.  You should only use this for writing low level
// setters/getters for SparseTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
//
// This may be called repeatedly, so make sure it's pretty cheap.
inline SparseTensorImpl* get_sparse_impl(const SparseTensor& self) {
  TORCH_INTERNAL_ASSERT(
      self.is_sparse(), "_internal_get_SparseTensorImpl: not a sparse tensor");
  return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

// Takes indices and values and directly puts them into the sparse tensor, no
// copy.  This used to be called THSTensor_(_move)
inline void alias_into_sparse(
    const SparseTensor& self,
    const Tensor& indices,
    const Tensor& values) {
  get_sparse_impl(self)->set_indices_and_values_unsafe(indices, values);
}

// Take indices and values and makes a (data) copy of them to put into the
// sparse indices/values.  This used to be called THSTensor_(_set)
inline void copy_into_sparse(
    const SparseTensor& self,
    const Tensor& indices,
    const Tensor& values,
    bool non_blocking) {
  alias_into_sparse(
      self,
      indices.to(self._indices().options(), non_blocking, /*copy=*/true),
      values.to(self._values().options(), non_blocking, /*copy=*/true));
}

// TODO: put this into the public API
inline bool is_same_tensor(const Tensor& lhs, const Tensor& rhs) {
  return lhs.unsafeGetTensorImpl() == rhs.unsafeGetTensorImpl();
}

inline bool is_same_density(const SparseTensor& self, const SparseTensor& src) {
  return self.sparse_dim() == src.sparse_dim() &&
      self.dense_dim() == src.dense_dim();
}

// Give us a new values tensor, with the same dimensionality
// as 'values' but with a new number of non-zero elements.
// TODO: Expose this for real in ATen, some day?
// NB: Doesn't preserve data.
inline Tensor new_values_with_size_of(const Tensor& values, int64_t nnz) {
  std::vector<int64_t> size = values.sizes().vec();
  size[0] = nnz;
  return at::empty(size, values.options());
}

// NOTE [ Flatten Sparse Indices ]
// This helper function flattens a sparse indices tensor (a Tensor) into a 1D
// indices tensor. E.g.,
//   input = [[2, 4, 0],
//            [3, 1, 10]]
//   full_size = [2, 12]
//   output = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 10 ] = [27, 49, 10]
//
// In other words, assuming that each `indices[i, :]` is a valid index to a
// tensor `t` of shape `full_size`. This returns the corresponding indices to
// the flattened tensor `t.reshape( prod(full_size[:indices.size(0)]), -1 )`.
// if forceClone is true, the result will forced to be a clone of self.
// if force_clone is true, the result will forced to be a clone of self.
TORCH_API Tensor flatten_indices(
    const Tensor& indices,
    IntArrayRef full_size,
    bool force_clone = false);

// Flatten sparse tensor's indices from nD to 1D, similar to NOTE [ Flatten
// Sparse Indices ], except this one allows partial flatten: only flatten on
// specified dims. Note that the flatten indices might be uncoalesced if
// dims_to_flatten.size() < sparse_dim. Also if input indices is already
// coalesced, the flattened indices will also be sorted.
//
// args:
//    indices: sparse tensor indices
//    sizes: sparse tensor sizes
//    dims_to_flatten: a list of dim index to flatten
//
// Ex1:
//   indices = [[2, 4, 0],
//             [3, 1, 3]]
//   sizes = [2, 12]
//   dims_to_flatten = [0, 1]
//   new_indices = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 3 ] = [27, 49, 3]
//
// Ex2:
//   dims_to_flatten = [1]
//   new_indices = [ 3, 1, 3 ]  # uncoalesced
TORCH_API Tensor flatten_indices_by_dims(
    const Tensor& indices,
    const IntArrayRef& sizes,
    const IntArrayRef& dims_to_flatten);

// Find the CSR representation for a row `indices` from the COO format
TORCH_API Tensor coo_to_csr(const int64_t* indices, int64_t dim, int64_t nnz);

TORCH_API Tensor zeros_like_with_indices(const Tensor& t);

template <size_t static_shape_max_len>
class TensorGeometryHolder {
  using geometry_holder_t = std::array<int64_t, static_shape_max_len>;

 public:
  explicit TensorGeometryHolder(
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options = {}) {
    std::copy(sizes.begin(), sizes.end(), t_sizes.begin());
    std::copy(strides.begin(), strides.end(), t_strides.begin());
  }

  explicit TensorGeometryHolder(const Tensor& t)
      : TensorGeometryHolder(t.sizes(), t.strides()) {}

  auto operator*() const {
    return std::make_tuple(t_sizes, t_strides);
  }

 private:
  geometry_holder_t t_sizes;
  geometry_holder_t t_strides;
};

template <>
class TensorGeometryHolder<0> {
  using geometry_holder_t = Tensor;

 public:
  explicit TensorGeometryHolder(
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options) {
    const int64_t t_ndims = sizes.size();
    const auto cpu_options = TensorOptions(options).dtype(kLong).device(kCPU);
    Tensor t_sizes_and_strides_cpu = at::empty({2, t_ndims}, cpu_options);
    t_sizes_and_strides_cpu.select(0, 0).copy_(at::tensor(sizes, cpu_options));
    t_sizes_and_strides_cpu.select(0, 1).copy_(
        at::tensor(strides, cpu_options));
    const Tensor t_sizes_and_strides =
        t_sizes_and_strides_cpu.to(options.device());
    t_sizes = t_sizes_and_strides.select(0, 0);
    t_strides = t_sizes_and_strides.select(0, 1);
  }

  explicit TensorGeometryHolder(const Tensor& t)
      : TensorGeometryHolder(t.sizes(), t.strides(), t.options()) {}

  auto operator*() const {
    return std::make_tuple(
        t_sizes.template data_ptr<int64_t>(),
        t_strides.template data_ptr<int64_t>());
  }

 private:
  geometry_holder_t t_sizes;
  geometry_holder_t t_strides;
};

// Return all indices of a tensor with the given shape.
//
// full_coo_indices(shape) is equivalent to
// torch.ones(shape).nonzero().transpose(-2, -1) but much faster.
TORCH_API Tensor full_coo_indices(IntArrayRef sizes, TensorOptions options);

} // namespace at::sparse

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TensorGeometryHolder`, `TensorGeometryHolder`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Parallel.h`
- `ATen/SparseTensorImpl.h`
- `ATen/core/Tensor.h`
- `ATen/Functions.h`
- `ATen/ops/empty.h`
- `ATen/ops/tensor.h`


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

- **File Documentation**: `SparseTensorUtils.h_docs.md`
- **Keyword Index**: `SparseTensorUtils.h_kw.md`
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

- **File Documentation**: `SparseTensorUtils.h_docs.md_docs.md`
- **Keyword Index**: `SparseTensorUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
