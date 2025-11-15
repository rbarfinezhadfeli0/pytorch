# Documentation: `docs/torch/csrc/inductor/aoti_runtime/thread_local.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_runtime/thread_local.h_docs.md`
- **Size**: 7,152 bytes (6.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_runtime/thread_local.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_runtime/thread_local.h`
- **Size**: 4,352 bytes (4.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>

namespace torch::aot_inductor {

template <typename T>
struct ThreadLocalCachedOutputTensor;

template <>
struct ThreadLocalCachedOutputTensor<RAIIAtenTensorHandle> {
  explicit ThreadLocalCachedOutputTensor(const RAIIAtenTensorHandle&) {}
  void copy_data_from(const RAIIAtenTensorHandle& handle) {
    throw std::runtime_error("can't happen");
  }

  AtenTensorHandle tensor() const {
    throw std::runtime_error("can't happen");
  }
};

template <>
struct ThreadLocalCachedOutputTensor<AtenTensorHandle> {
  explicit ThreadLocalCachedOutputTensor(const AtenTensorHandle&) {}
  void copy_data_from(const AtenTensorHandle& handle) {
    throw std::runtime_error("can't happen");
  }

  AtenTensorHandle tensor() const {
    throw std::runtime_error("can't happen");
  }
};

template <>
struct ThreadLocalCachedOutputTensor<ConstantHandle> {
  explicit ThreadLocalCachedOutputTensor(const ConstantHandle&) {}
  void copy_data_from(const ConstantHandle& handle) {
    throw std::runtime_error("can't happen");
  }

  AtenTensorHandle tensor() const {
    throw std::runtime_error("can't happen");
  }
};

template <typename T>
struct ThreadLocalCachedOutputTensor<ArrayRefTensor<T>> {
  explicit ThreadLocalCachedOutputTensor(const ArrayRefTensor<T>& t) {
    realloc(t);
  }

  void copy_data_from(const ArrayRefTensor<T>& t) {
    if (t.numel() > capacity_) {
      realloc(t);
    }
    std::copy(t.data(), t.data() + t.numel(), storage_.get());
  }

  AtenTensorHandle tensor() const {
    return tensor_.get();
  }

 private:
  void realloc(const ArrayRefTensor<T>& t) {
    capacity_ = t.numel();
    // NOLINTNEXTLINE(*arrays*)
    storage_ = std::make_unique<T[]>(t.numel());
    AtenTensorHandle handle = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
        storage_.get(),
        t.sizes().size(),
        t.sizes().data(),
        t.strides().data(),
        0,
        aoti_torch_dtype<std::remove_const_t<T>>(),
        t.device_type(),
        t.device_idx(),
        &handle));
    tensor_ = handle;
  }

  // NOLINTNEXTLINE(*arrays*)
  std::unique_ptr<T[]> storage_;
  int64_t capacity_ = 0;
  RAIIAtenTensorHandle tensor_;
};

template <typename T>
struct ThreadLocalCachedOutputArray;

// Just needs to compile, doesn't need to do anything.
template <>
struct ThreadLocalCachedOutputArray<RAIIAtenTensorHandle> {
  explicit ThreadLocalCachedOutputArray(const RAIIAtenTensorHandle&) {
    throw std::runtime_error("can't happen");
  }

  // Not supported yet! We would need to put contiguous() or
  // expect_contiguous() into the ABI.
  void copy_data_from(const RAIIAtenTensorHandle&) {
    throw std::runtime_error("can't happen");
  }

  template <typename U>
  ArrayRefTensor<U> arrayref_tensor() const {
    throw std::runtime_error("can't happen");
  }
};

// Just needs to compile, doesn't need to do anything.
template <>
struct ThreadLocalCachedOutputArray<ConstantHandle> {
  explicit ThreadLocalCachedOutputArray(const ConstantHandle&) {
    throw std::runtime_error("can't happen");
  }

  // Not supported yet! We would need to put contiguous() or
  // expect_contiguous() into the ABI.
  void copy_data_from(const ConstantHandle&) {
    throw std::runtime_error("can't happen");
  }

  template <typename U>
  ArrayRefTensor<U> arrayref_tensor() const {
    throw std::runtime_error("can't happen");
  }
};

template <typename T>
struct ThreadLocalCachedOutputArray<ArrayRefTensor<T>> {
  explicit ThreadLocalCachedOutputArray(const ArrayRefTensor<T>& t) {}

  template <
      typename U,
      std::enable_if_t<
          std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
          bool> = true>
  ArrayRefTensor<T> arrayref_tensor() const {
    return tensor_;
  }

  void copy_data_from(const ArrayRefTensor<T>& t) {
    if (t.numel() > capacity_) {
      capacity_ = t.numel();
      // NOLINTNEXTLINE(*arrays*)
      storage_ = std::make_unique<T[]>(capacity_);
    }
    std::copy(t.data(), t.data() + t.numel(), storage_.get());
    tensor_ = t;
    tensor_.set_arrayref(MiniArrayRef<T>(storage_.get(), t.numel()));
  }

 private:
  // NOLINTNEXTLINE(*arrays*)
  std::unique_ptr<T[]> storage_;
  uint32_t capacity_ = 0;
  ArrayRefTensor<T> tensor_;
};

} // namespace torch::aot_inductor

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ThreadLocalCachedOutputTensor`, `ThreadLocalCachedOutputTensor`, `ThreadLocalCachedOutputTensor`, `ThreadLocalCachedOutputTensor`, `ThreadLocalCachedOutputTensor`, `ThreadLocalCachedOutputArray`, `ThreadLocalCachedOutputArray`, `ThreadLocalCachedOutputArray`, `ThreadLocalCachedOutputArray`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/inductor/aoti_runtime/arrayref_tensor.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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
- [`scalar_to_tensor.h_docs.md`](./scalar_to_tensor.h_docs.md)
- [`model_base.h_docs.md`](./model_base.h_docs.md)
- [`mini_array_ref.h_docs.md`](./mini_array_ref.h_docs.md)
- [`device_utils.h_docs.md`](./device_utils.h_docs.md)
- [`model.h_docs.md`](./model.h_docs.md)


## Cross-References

- **File Documentation**: `thread_local.h_docs.md`
- **Keyword Index**: `thread_local.h_kw.md`
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
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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
- [`interface.h_docs.md_docs.md`](./interface.h_docs.md_docs.md)
- [`arrayref_tensor.h_docs.md_docs.md`](./arrayref_tensor.h_docs.md_docs.md)
- [`device_utils.h_kw.md_docs.md`](./device_utils.h_kw.md_docs.md)
- [`utils_xpu.h_docs.md_docs.md`](./utils_xpu.h_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`mini_array_ref.h_docs.md_docs.md`](./mini_array_ref.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `thread_local.h_docs.md_docs.md`
- **Keyword Index**: `thread_local.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
