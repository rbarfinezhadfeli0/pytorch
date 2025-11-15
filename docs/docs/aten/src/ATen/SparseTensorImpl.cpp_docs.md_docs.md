# Documentation: `docs/aten/src/ATen/SparseTensorImpl.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/SparseTensorImpl.cpp_docs.md`
- **Size**: 8,475 bytes (8.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/SparseTensorImpl.cpp`

## File Metadata

- **Path**: `aten/src/ATen/SparseTensorImpl.cpp`
- **Size**: 5,921 bytes (5.78 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

namespace {
  DeviceType sparseTensorSetToDeviceType(DispatchKeySet key_set) {
    auto k = c10::highestPriorityBackendTypeId(key_set);
    TORCH_CHECK(c10::toFunctionalityKey(k) == DispatchKey::Sparse,
      "cannot create sparse tensor with non sparse dispatch key ", k);
    return c10::dispatchKeyToDeviceType(k);
  }
}


// An empty dense tensor defaults to a 1-dimensional tensor of size [0]
// (recall, it is not a 0-dimensional tensor, because such a tensor would
// a scalar and have one element)
//
// Thus, an empty sparse tensor should be a 1-dimensional tensor of size [0].
// Furthermore, we have dim == sparse_dim + dense_dim; since this is a sparse
// tensor, let us say that an empty sparse tensor has sparse_dim == 1 and
// dense_dim == 0.  (There is a degree of freedom here, but given that this
// is a sparse dimension, it seems reasonable to demand that sparse_dim > 0).
//
// This means that we allocate a [1,0] size indices tensor and a [0] size
// values tensor for such an empty tensor.
SparseTensorImpl::SparseTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta data_type)
  :   SparseTensorImpl(key_set, data_type
      , at::empty({1, 0}, at::initialTensorOptions().device(sparseTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , at::empty({0}, at::initialTensorOptions().device(sparseTensorSetToDeviceType(key_set)).dtype(data_type))) {}

SparseTensorImpl::SparseTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta data_type, at::Tensor indices, at::Tensor values)
    : TensorImpl(key_set, data_type, values.device())
    , sparse_dim_(1)
    , indices_(std::move(indices))
    , values_(std::move(values)) {
  // we proxy to this constructor so we can initialize the device correctly, but really only indices/values of this shape are allowed.
  AT_ASSERT(indices_.sizes() == IntArrayRef({1, 0}));
  AT_ASSERT(values_.sizes() == IntArrayRef({0}));
  AT_ASSERT(values_.device() == indices_.device());
  AT_ASSERT(values_.device() == device());

  is_non_overlapping_and_dense_ = false;
  set_storage_access_should_throw();
  set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
}

  // Destructor doesn't call release_resources because it's
  // unnecessary; don't forget to change that if needed!
void SparseTensorImpl::release_resources() {
  TensorImpl::release_resources();
  values_.reset();
  indices_.reset();
}

void SparseTensorImpl::set_size(int64_t dim, int64_t new_size) {
  TORCH_CHECK(false, "sparse tensors do not have set_size");
}
void SparseTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_CHECK(false, "sparse tensors do not have set_stride");
}
void SparseTensorImpl::set_storage_offset(int64_t storage_offset) {
  TORCH_CHECK(false, "sparse tensors do not have set_storage_offset");
}
#ifdef DEBUG
bool SparseTensorImpl::has_storage() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "SparseTensorImpl assumes that storage_ is never set");
  return false;
}
#endif

const char* SparseTensorImpl::tensorimpl_type_name() const {
  return "SparseTensorImpl";
}

void SparseTensorImpl::set_indices_and_values_unsafe(const Tensor& indices, const Tensor& values) {
  TORCH_CHECK(allow_tensor_metadata_change(), "set_indices_and_values_unsafe ", err_msg_tensor_metadata_change_not_allowed);

  TORCH_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());
  TORCH_CHECK(!values.is_sparse(), "expected values to be a dense tensor, but got values of layout ", values.layout());

  TORCH_CHECK(values.device().type() == device().type(), "device type of values (", values.device().type(), ") must match device type of device().type()", device().type(), ")");
  TORCH_CHECK(values.scalar_type() == typeMetaToScalarType(dtype()), "dtype of values (", values.scalar_type(), ") must match dtype of sparse tensor (", typeMetaToScalarType(dtype()), ")");
  TORCH_CHECK(indices.scalar_type() == kLong, "indices must be an int64 tensor");
  TORCH_CHECK(indices.options().backend() == values.options().backend(), "backend of indices (", indices.options().backend(), ") must match backend of values (", values.options().backend(), ")");
  TORCH_CHECK(!indices.is_cuda() || indices.get_device() == values.get_device(), "device of indices (", indices.get_device(), ") must match device of values (", values.get_device(), ")");

  TORCH_CHECK(indices.dim() == 2, "indices must be sparse_dim x nnz, but got: ", indices.sym_sizes());
  TORCH_CHECK(indices.sym_size(1) == values.sym_size(0), "indices and values must have same nnz, but got nnz from indices: ", indices.sym_size(1), ", nnz from values: ", values.sym_size(0));
  TORCH_CHECK(indices.sym_size(0) == sparse_dim_, "indices has incorrect first dimension, expected ", sparse_dim_, ", got ", indices.sym_size(0));
  TORCH_CHECK(values.dim() == dense_dim_ + 1, "values has incorrect number of dimensions, expected ", dense_dim_ + 1, ", got ", values.dim());

  auto dense_size_original = sym_sizes().slice(sparse_dim_);
  std::vector<c10::SymInt> expected_values_size_vec = {values.sym_size(0)};
  expected_values_size_vec.insert(expected_values_size_vec.end(), dense_size_original.begin(), dense_size_original.end());
  SymIntArrayRef expected_values_size(expected_values_size_vec);
  auto new_values_size = values.sym_sizes();
  TORCH_CHECK(
    std::equal(expected_values_size.begin(), expected_values_size.end(), new_values_size.begin()),
    "values has incorrect size, expected ", expected_values_size, ", got ", new_values_size
  );

  indices_ = indices;
  values_ = values;
  AT_ASSERT(device() == values_.device());
  AT_ASSERT(values_.device() == indices_.device());

  coalesced_ = TORCH_GUARD_OR_FALSE(sym_nnz().sym_lt(2));
}


} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

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

- `ATen/ATen.h`
- `ATen/SparseTensorImpl.h`
- `ATen/InitialTensorOptions.h`
- `ATen/core/LegacyTypeDispatch.h`


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

- **File Documentation**: `SparseTensorImpl.cpp_docs.md`
- **Keyword Index**: `SparseTensorImpl.cpp_kw.md`
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

- **File Documentation**: `SparseTensorImpl.cpp_docs.md_docs.md`
- **Keyword Index**: `SparseTensorImpl.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
