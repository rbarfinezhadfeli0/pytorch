# Documentation: `aten/src/ATen/OpaqueTensorImpl.h`

## File Metadata

- **Path**: `aten/src/ATen/OpaqueTensorImpl.h`
- **Size**: 6,815 bytes (6.66 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/MemoryFormat.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

// An "Opaque" TensorImpl -- there are no strides and (for now)
// even data() is not supported (thus no pointer arithmetic).

// NOTE: We could allow data() in the future, but would have to ensure pointer
// arithmetic code is properly guarded.
//
// NOTE: This does not support resize_ (and other metadata-changing ops) because
// of `shallow_copy_and_detach`. We would need to define an interface to
// "shallow copy" in order to add support.

template <typename OpaqueHandle>
struct TORCH_API OpaqueTensorImpl : public TensorImpl {
  // public constructor for now...
  OpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      bool is_non_overlapping_and_dense = true)
      : TensorImpl(key_set, data_type, device),
        opaque_handle_(std::move(opaque_handle)) {
    constructor_impl(sizes, is_non_overlapping_and_dense);
  }

  OpaqueTensorImpl(
      TensorImpl::ImplType impl_type,
      c10::Storage&& storage,
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      bool is_non_overlapping_and_dense = true)
      : TensorImpl(impl_type, std::move(storage), key_set, data_type),
        opaque_handle_(std::move(opaque_handle)) {
    constructor_impl(sizes, is_non_overlapping_and_dense);
  }

  // Destructor doesn't call release_resources because it's
  // unnecessary; don't forget to change that if needed!
  void release_resources() override {
    TensorImpl::release_resources();
    opaque_handle_ = {};
  }

  void set_size(int64_t dim, int64_t new_size) override {
    TORCH_CHECK(false, "opaque tensors do not have set_size");
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    TORCH_CHECK(false, "opaque tensors do not have set_stride");
  }

  void set_storage_offset(int64_t storage_offset) override {
    TORCH_CHECK(false, "opaque tensors do not have set_storage_offset");
  }

#ifdef DEBUG
  bool has_storage() const override {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !storage_, "OpaqueTensorImpl assumes that storage_ is never set");
    return false;
  }
#endif

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
        key_set(),
        dtype(),
        device(),
        opaque_handle_,
        sizes_and_strides_.sizes_arrayref());
    copy_tensor_metadata(
        /*src_opaque_impl=*/this,
        /*dest_opaque_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
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
    auto impl = c10::make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
        key_set(),
        dtype(),
        device(),
        opaque_handle_,
        sizes_and_strides_.sizes_arrayref());
    copy_tensor_metadata(
        /*src_opaque_impl=*/this,
        /*dest_opaque_impl=*/impl.get(),
        /*version_counter=*/std::move(version_counter),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's
   * `allow_tensor_metadata_change_`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    auto opaque_impl =
        static_cast<const OpaqueTensorImpl<OpaqueHandle>*>(impl.get());
    copy_tensor_metadata(
        /*src_impl=*/opaque_impl,
        /*dest_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
  }

  const OpaqueHandle& opaque_handle() const {
    return opaque_handle_;
  }

  OpaqueHandle& unsafe_opaque_handle() {
    return opaque_handle_;
  }

 protected:
  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const OpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
      OpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_opaque_impl,
        dest_opaque_impl,
        version_counter,
        allow_tensor_metadata_change);

    // OpaqueTensorImpl-specific fields.
    dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
  }

  static void copy_tensor_metadata(
      const OpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
      OpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_opaque_impl,
        dest_opaque_impl,
        std::move(version_counter),
        allow_tensor_metadata_change);

    // OpaqueTensorImpl-specific fields.
    dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
  }

 private:
  const char* tensorimpl_type_name() const override {
    return "OpaqueTensorImpl";
  }

  void constructor_impl(
      c10::IntArrayRef sizes,
      bool is_non_overlapping_and_dense) {
    set_storage_access_should_throw();
    set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
    sizes_and_strides_.set_sizes(sizes);
    refresh_numel();
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    is_non_overlapping_and_dense_ = is_non_overlapping_and_dense;
  }

  OpaqueHandle opaque_handle_;
};

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

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

- `c10/core/MemoryFormat.h`
- `c10/core/SymIntArrayRef.h`
- `c10/core/TensorImpl.h`
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

- **File Documentation**: `OpaqueTensorImpl.h_docs.md`
- **Keyword Index**: `OpaqueTensorImpl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
