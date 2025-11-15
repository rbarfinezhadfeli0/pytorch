# Documentation: `docs/torch/csrc/inductor/aoti_runtime/arrayref_tensor.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_runtime/arrayref_tensor.h_docs.md`
- **Size**: 8,791 bytes (8.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_runtime/arrayref_tensor.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_runtime/arrayref_tensor.h`
- **Size**: 6,256 bytes (6.11 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/inductor/aoti_runtime/mini_array_ref.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace torch::aot_inductor {

using MiniIntArrayRef = MiniArrayRef<int64_t>;

static_assert(
    sizeof(MiniIntArrayRef) == sizeof(void*) + sizeof(size_t),
    "changing the size of MiniArrayRef breaks ABI compatibility!");

inline bool is_contiguous_strides_for_shape(
    int64_t ndim,
    const int64_t* strides_ptr,
    const int64_t* sizes_ptr) {
  int64_t z = 1;
  for (int64_t d = ndim - 1; d >= 0; d--) {
    const auto& size_d = sizes_ptr[d];
    if (size_d != 1) {
      if (strides_ptr[d] == z) {
        z *= size_d;
      } else {
        return false;
      }
    }
  }
  return true;
}

// Shim for AOTI generated code to pretend a raw array works like an
// AtenTensorHandle.
template <typename T>
class ArrayRefTensor {
 public:
  ArrayRefTensor() = default;

  explicit ArrayRefTensor(
      MiniArrayRef<T> arr,
      MiniArrayRef<const int64_t> sizes,
      MiniArrayRef<const int64_t> strides,
      int32_t device_type,
      int32_t device_idx)
      : arrayRef_(arr),
        sizes_(sizes),
        strides_(strides),
        device_type_(device_type),
        device_idx_(device_idx) {
    assert(sizes.size() == strides.size());
    assert(is_contiguous_strides_for_shape(
        sizes.size(), strides.data(), sizes.data()));
  }

  AtenTensorHandle expensiveCopyToTensor() const {
    AtenTensorHandle result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
        sizes_.size(),
        sizes_.data(),
        strides_.data(),
        aoti_torch_dtype<std::remove_const_t<T>>(),
        device_type_,
        device_idx_,
        &result));
    void* dataPtr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(result, &dataPtr));
    std::memcpy(dataPtr, data(), numel() * sizeof(T));
    return result;
  }

  // We need to look the same as RAIIAtenTensorHandle, which returns
  // an owning AtenTensorHandle from release(). So, we allocate one!
  AtenTensorHandle release() {
    return expensiveCopyToTensor();
  }

  AtenTensorHandle borrowAsTensor() const {
    AtenTensorHandle result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob_v2(
        data(),
        sizes_.size(),
        sizes_.data(),
        strides_.data(),
        0,
        aoti_torch_dtype<std::remove_const_t<T>>(),
        device_type_,
        device_idx_,
        &result,
        aoti_torch_layout_strided(),
        nullptr,
        0));
    return result;
  }

  // We don't need to free any memory.
  void reset() {}

  auto sizes() const {
    return sizes_;
  }

  auto strides() const {
    return strides_;
  }

  auto device_type() const {
    return device_type_;
  }

  auto device_idx() const {
    return device_idx_;
  }

  T* data() const {
    return arrayRef_.data();
  }

  auto numel() const {
    return arrayRef_.size();
  }

  void set_arrayref(MiniArrayRef<T> new_arrayref) {
    arrayRef_ = new_arrayref;
  }

 private:
  MiniArrayRef<T> arrayRef_;
  // We expect generated code to have statically available sizes &
  // strides for us.
  MiniArrayRef<const int64_t> sizes_;
  MiniArrayRef<const int64_t> strides_;
  int32_t device_type_ = 0;
  int32_t device_idx_ = 0;
  // We continue to zero-initialize this field in case we repurpose
  // the space later; having predictable contents can only help.
  int32_t unusedDoNotRemoveForABICompatibility_ = 0;
};

static_assert(
    sizeof(ArrayRefTensor<int>) ==
        3 * sizeof(MiniIntArrayRef) + 3 * sizeof(int32_t) +
            (alignof(ArrayRefTensor<int>) > 4 ? sizeof(int32_t) : 0),
    "changing the size of ArrayRefTensor breaks ABI compatibility!");

template <typename T>
inline ArrayRefTensor<T> reinterpret_tensor_wrapper(
    const ArrayRefTensor<T>& self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset) {
  // REVIEW: we should add a way to build the DSO in debug mode during
  // tests so we can have checks like this!
  assert(is_contiguous_strides_for_shape(ndim, strides_ptr, sizes_ptr));
  return ArrayRefTensor<T>(
      MiniArrayRef<T>(
          self.data() + storage_offset, self.numel() - storage_offset),
      MiniArrayRef<const int64_t>(sizes_ptr, ndim),
      MiniArrayRef<const int64_t>(strides_ptr, ndim),
      self.device_type(),
      self.device_idx());
}

template <typename T>
inline T* get_data_ptr_wrapper(ArrayRefTensor<T>& tensor) {
  return tensor.data();
}

template <typename T>
inline T* get_data_ptr_wrapper(const MiniArrayRef<T>& arr) {
  return arr.data();
}

template <typename T>
inline const ArrayRefTensor<T>& unwrap_raii_handle_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor;
}

template <typename T>
inline ArrayRefTensor<T>& unwrap_raii_handle_if_needed(
    ArrayRefTensor<T>& tensor) {
  return tensor;
}

template <typename T>
inline const ArrayRefTensor<T>& wrap_with_raii_handle_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor;
}

template <typename T>
inline ArrayRefTensor<T>& wrap_with_raii_handle_if_needed(
    ArrayRefTensor<T>& tensor) {
  return tensor;
}

template <typename T>
inline ArrayRefTensor<T> wrap_with_raii_handle_if_needed(
    ArrayRefTensor<T>&& tensor) {
  return std::move(tensor);
}

template <typename T>
inline RAIIAtenTensorHandle expensive_copy_to_tensor_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor.expensiveCopyToTensor();
}

inline AtenTensorHandle expensive_copy_to_tensor_if_needed(
    AtenTensorHandle handle) {
  return handle;
}

template <typename T>
const T& copy_arrayref_tensor_to_tensor(const T& t) {
  return t;
}

template <typename T>
RAIIAtenTensorHandle copy_arrayref_tensor_to_tensor(
    const ArrayRefTensor<T>& art) {
  return art.expensiveCopyToTensor();
}

template <typename T>
const T& borrow_arrayref_tensor_as_tensor(const T& t) {
  return t;
}

template <typename T>
RAIIAtenTensorHandle borrow_arrayref_tensor_as_tensor(
    const ArrayRefTensor<T>& art) {
  return art.borrowAsTensor();
}

} // namespace torch::aot_inductor

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ArrayRefTensor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/inductor/aoti_runtime/mini_array_ref.h`
- `torch/csrc/inductor/aoti_runtime/utils.h`
- `torch/csrc/inductor/aoti_torch/c/shim.h`
- `cassert`
- `cstdint`
- `cstring`


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

Files in the same folder (`torch/csrc/inductor/aoti_runtime`):

- [`sycl_runtime_wrappers.h_docs.md`](./sycl_runtime_wrappers.h_docs.md)
- [`utils_xpu.h_docs.md`](./utils_xpu.h_docs.md)
- [`utils_cuda.h_docs.md`](./utils_cuda.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`thread_local.h_docs.md`](./thread_local.h_docs.md)
- [`scalar_to_tensor.h_docs.md`](./scalar_to_tensor.h_docs.md)
- [`model_base.h_docs.md`](./model_base.h_docs.md)
- [`mini_array_ref.h_docs.md`](./mini_array_ref.h_docs.md)
- [`device_utils.h_docs.md`](./device_utils.h_docs.md)
- [`model.h_docs.md`](./model.h_docs.md)


## Cross-References

- **File Documentation**: `arrayref_tensor.h_docs.md`
- **Keyword Index**: `arrayref_tensor.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/inductor/aoti_runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/inductor/aoti_runtime`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/inductor/aoti_runtime`):

- [`constant_type.h_kw.md_docs.md`](./constant_type.h_kw.md_docs.md)
- [`sycl_runtime_wrappers.h_kw.md_docs.md`](./sycl_runtime_wrappers.h_kw.md_docs.md)
- [`arrayref_tensor.h_kw.md_docs.md`](./arrayref_tensor.h_kw.md_docs.md)
- [`thread_local.h_docs.md_docs.md`](./thread_local.h_docs.md_docs.md)
- [`interface.h_docs.md_docs.md`](./interface.h_docs.md_docs.md)
- [`device_utils.h_kw.md_docs.md`](./device_utils.h_kw.md_docs.md)
- [`utils_xpu.h_docs.md_docs.md`](./utils_xpu.h_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`mini_array_ref.h_docs.md_docs.md`](./mini_array_ref.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `arrayref_tensor.h_docs.md_docs.md`
- **Keyword Index**: `arrayref_tensor.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
