# Documentation: `docs/aten/src/ATen/native/mps/TensorFactory.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mps/TensorFactory.cpp_docs.md`
- **Size**: 7,975 bytes (7.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mps/TensorFactory.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mps/TensorFactory.cpp`
- **Size**: 5,549 bytes (5.42 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <torch/library.h>
#include <ATen/mps/EmptyTensor.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/TensorFactory.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#endif
#include <ATen/ops/_efficientzerotensor_native.h>

#include <utility>

namespace at::native {

static inline void maybe_resize_storage_mps(TensorImpl* self, uint64_t new_size) {
  if (new_size == 0) {
    return;
  }

  auto storage = self->storage().unsafeGetStorageImpl();
  if (!storage) {
    TORCH_CHECK(false, "Tensor: invalid null storage");
  }
  uint64_t new_size_bytes = (new_size + self->storage_offset()) * self->dtype().itemsize();
  if (new_size_bytes > self->storage().nbytes()) {
    if (new_size_bytes == 0) {
      storage->set_data_ptr_noswap(at::DataPtr(nullptr, at::Device(at::DeviceType::MPS, 0)));
      storage->set_nbytes(0);
    } else {
      at::DataPtr new_data = storage->allocator()->allocate(new_size_bytes);
      size_t copy_capacity = std::min<size_t>(new_size_bytes, storage->nbytes());
      if (storage->data() && copy_capacity > 0) {
        at::native::mps::copy_blit_mps(new_data.get(), storage->data(), copy_capacity);
      }
      // Destructively overwrite data_ptr
      storage->set_data_ptr_noswap(std::move(new_data));
      storage->set_nbytes(new_size_bytes);
    }
  }
}

inline TensorImpl* resize_impl_mps_(
    TensorImpl* self,
    IntArrayRef size,
    std::optional<IntArrayRef> stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    storage_size = storage_size_for(size, *stride);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_mps(self, storage_size);

  return self;
}

Tensor empty_mps(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {

  return at::detail::empty_mps(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

Tensor empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  check_size_nonnegative(size);
  // empty memory formatempty
  auto t = at::native::empty_mps(
      {0},
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt);
  resize_impl_mps_(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

const Tensor& resize_mps_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  resize_impl_mps_(self_, size, /*stride=*/std::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }
  return self;
}

Tensor& set_mps_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      at::mps::GetMPSAllocator(),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor& set_storage_mps_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  checkSetStorage(result, std::move(storage), storage_offset, size, stride);
  //std::cout << "set storage_mps " << storage_offset << " stride " << stride << std::endl;
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  std::optional<IntArrayRef> stride_opt = stride.data() != nullptr ?
                                          std::optional<IntArrayRef>(stride) : std::nullopt;
  at::native::resize_impl_mps_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

Tensor _efficientzerotensor_mps(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
    auto device_ = device_or_default(device);
    auto allocator = at::native::ZeroTensorAllocator(device_);
    auto dtype_ = dtype_or_default(dtype);
    auto zero_ks = at::DispatchKeySet(c10::DispatchKey::MPS) | at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
    auto out = at::detail::empty_generic(size, &allocator, zero_ks, dtype_, std::nullopt);
    return out;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mps`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/Tensor.h`
- `ATen/Utils.h`
- `torch/library.h`
- `ATen/mps/EmptyTensor.h`
- `ATen/mps/MPSDevice.h`
- `ATen/native/Resize.h`
- `ATen/native/ResizeCommon.h`
- `ATen/native/mps/Copy.h`
- `ATen/native/mps/TensorFactory.h`
- `ATen/Dispatch.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_efficientzerotensor_native.h`
- `utility`


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

Files in the same folder (`aten/src/ATen/native/mps`):

- [`OperationUtils.h_docs.md`](./OperationUtils.h_docs.md)
- [`TensorFactory.h_docs.md`](./TensorFactory.h_docs.md)
- [`Copy.h_docs.md`](./Copy.h_docs.md)
- [`MPSGraphSequoiaOps.h_docs.md`](./MPSGraphSequoiaOps.h_docs.md)
- [`MetalShaderLibrary.h_docs.md`](./MetalShaderLibrary.h_docs.md)


## Cross-References

- **File Documentation**: `TensorFactory.cpp_docs.md`
- **Keyword Index**: `TensorFactory.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/mps`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/mps`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/mps`):

- [`TensorFactory.h_kw.md_docs.md`](./TensorFactory.h_kw.md_docs.md)
- [`OperationUtils.h_kw.md_docs.md`](./OperationUtils.h_kw.md_docs.md)
- [`TensorFactory.cpp_kw.md_docs.md`](./TensorFactory.cpp_kw.md_docs.md)
- [`MPSGraphSequoiaOps.h_kw.md_docs.md`](./MPSGraphSequoiaOps.h_kw.md_docs.md)
- [`OperationUtils.h_docs.md_docs.md`](./OperationUtils.h_docs.md_docs.md)
- [`MPSGraphSequoiaOps.h_docs.md_docs.md`](./MPSGraphSequoiaOps.h_docs.md_docs.md)
- [`Copy.h_kw.md_docs.md`](./Copy.h_kw.md_docs.md)
- [`MetalShaderLibrary.h_docs.md_docs.md`](./MetalShaderLibrary.h_docs.md_docs.md)
- [`MetalShaderLibrary.h_kw.md_docs.md`](./MetalShaderLibrary.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `TensorFactory.cpp_docs.md_docs.md`
- **Keyword Index**: `TensorFactory.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
