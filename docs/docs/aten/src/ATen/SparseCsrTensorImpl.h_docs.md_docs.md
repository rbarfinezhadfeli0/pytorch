# Documentation: `docs/aten/src/ATen/SparseCsrTensorImpl.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/SparseCsrTensorImpl.h_docs.md`
- **Size**: 9,686 bytes (9.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/SparseCsrTensorImpl.h`

## File Metadata

- **Path**: `aten/src/ATen/SparseCsrTensorImpl.h`
- **Size**: 7,167 bytes (7.00 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/Exception.h>
namespace at {

// Struct implementing a sparse CSR tensor. It uses three 1-D tensors for
// denoting the data: `crow_indices_`, `col_indices_` and `values_`.
// The `crow_indices_` tensor is a integer tensor of shape `(size(0) + 1)`
// that represents the compressed row indices of the CSR tensor. The
// `col_indices_` tensor is an integer tensor of shape `(nnz())`
// that explicitly stores the column indices of each value of the sparse
// tensor. The `values_` tensor can be of any pytorch-supported data type
// and has shape `(nnz())`.
//
// Since the main advantage of the CSR format over the COO format is speed of
// computation, care must be taken to facilitate smooth interfacing of
// these data structures with optimized libraries such as MKL and MAGMA.
// Since the MKL interface for pytorch currently uses indexing with int32
// type, it is important to make sure that the `crow_indices` and `col_indices`
// are of type int32 when calling MKL routines such as SPMM or SPMV.
//
// If not calling MKL, it should be alright to use 64 bit integer tensors
// for indexing.
struct TORCH_API SparseCsrTensorImpl : public TensorImpl {
  Tensor crow_indices_;
  Tensor col_indices_;
  Tensor values_;
  Layout layout_;

 public:
  explicit SparseCsrTensorImpl(
      at::DispatchKeySet /*key_set*/,
      at::Device device,
      Layout layout,
      const caffe2::TypeMeta /*data_type*/);

  void resize_(int64_t nnz, IntArrayRef size);
  void resize_and_clear_(
      int64_t sparse_dim,
      int64_t dense_dim,
      IntArrayRef size);
  void resize_as_sparse_compressed_tensor_(const Tensor& src);
  void set_member_tensors(
      const Tensor& crow_indices,
      const Tensor& col_indices,
      const Tensor& values,
      c10::SymIntArrayRef size);
  void set_member_tensors(
      const Tensor& crow_indices,
      const Tensor& col_indices,
      const Tensor& values,
      IntArrayRef size);
  const Tensor& compressed_indices() const {
    return crow_indices_;
  }
  const Tensor& plain_indices() const {
    return col_indices_;
  }
  const Tensor& values() const {
    return values_;
  }
  int64_t nnz() {
    return col_indices_.size(-1);
  }

  inline int64_t batch_dim() const noexcept {
    return crow_indices_.dim() - 1;
  }

  inline int64_t sparse_dim() const noexcept {
    return 2;
  }

  inline int64_t dense_dim() const noexcept {
    return values_.dim() - batch_dim() - block_dim() - 1;
  }

 private:
  inline int64_t block_dim() const noexcept {
    return (layout_ == kSparseBsr || layout_ == kSparseBsc ? 2 : 0);
  }

 protected:
  IntArrayRef strides_custom() const override;
  SymIntArrayRef sym_strides_custom() const override;
  SymBool sym_is_contiguous_custom(
      MemoryFormat /*memory_format*/) const override;

 public:
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;
  Layout layout_impl() const override {
    return layout_;
  }
  void set_layout(Layout layout) {
    switch (layout) {
      case kSparseCsr:
      case kSparseCsc:
      case kSparseBsr:
      case kSparseBsc:
        layout_ = layout;
        break;
      default:
        TORCH_CHECK(false, "unsupported layout ", layout);
    }
  }

  template <typename VariableVersion>
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
      VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const {
    const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
    c10::impl::PyInterpreter&& interpreter = nullptr;
    if (mode_stack_len > 0 &&
        !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
      const auto& cur_torch_dispatch_mode_state =
          c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
      interpreter = cur_torch_dispatch_mode_state->pyinterpreter();
    } else if (
        key_set_.has(DispatchKey::Python) &&
        !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
      interpreter = pyobj_slot_.load_pyobj_interpreter();
    } else {
      // otherwise just copy the SparseTensorImpl and not the PyObject.
      auto impl = c10::make_intrusive<SparseCsrTensorImpl>(
          key_set(), device(), layout_impl(), dtype());
      copy_tensor_metadata(
          /*src_sparse_impl=*/this,
          /*dest_sparse_impl=*/impl.get(),
          /*version_counter=*/version_counter,
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      impl->refresh_numel();
      return impl;
    }
    auto r = interpreter->detach(this);
    r->set_version_counter(std::forward<VariableVersion>(version_counter));
    r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    return r;
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(
        version_counter, allow_tensor_metadata_change);
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(
        std::move(version_counter), allow_tensor_metadata_change);
  }

 private:
  explicit SparseCsrTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      at::Tensor crow_indices,
      at::Tensor col_indices,
      at::Tensor values,
      at::Layout layout);

  const char* tensorimpl_type_name() const override;

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const SparseCsrTensorImpl* src_sparse_impl,
      SparseCsrTensorImpl* dest_sparse_impl,
      c10::VariableVersion version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_sparse_impl,
        dest_sparse_impl,
        std::move(version_counter),
        allow_tensor_metadata_change);

    // Sparse-specific fields
    dest_sparse_impl->crow_indices_ = src_sparse_impl->compressed_indices();
    dest_sparse_impl->col_indices_ = src_sparse_impl->plain_indices();
    dest_sparse_impl->values_ = src_sparse_impl->values();
    dest_sparse_impl->layout_ = src_sparse_impl->layout_impl();
  }
};
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 25 function(s).

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

- `ATen/Tensor.h`
- `c10/core/TensorImpl.h`
- `c10/core/impl/TorchDispatchModeTLS.h`
- `c10/util/Exception.h`


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

- **File Documentation**: `SparseCsrTensorImpl.h_docs.md`
- **Keyword Index**: `SparseCsrTensorImpl.h_kw.md`
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

- **File Documentation**: `SparseCsrTensorImpl.h_docs.md_docs.md`
- **Keyword Index**: `SparseCsrTensorImpl.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
