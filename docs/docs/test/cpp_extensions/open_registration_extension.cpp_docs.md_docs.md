# Documentation: `docs/test/cpp_extensions/open_registration_extension.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/open_registration_extension.cpp_docs.md`
- **Size**: 9,817 bytes (9.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/open_registration_extension.cpp`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension.cpp`
- **Size**: 6,808 bytes (6.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/view.h>

#include <unordered_map>

static uint64_t add_counter = 0;
static uint64_t last_saved_value = 0;

static uint64_t storageImpl_counter = 0;
static uint64_t last_storageImpl_saved_value = 0;

// A dummy storageImpl for our custom device, that secretly uses the CPU
c10::intrusive_ptr<c10::StorageImpl> make_custom_storage_impl(c10::StorageImpl::use_byte_size_t,
                                                              c10::SymInt size_bytes,
                                                              c10::DataPtr data_ptr,
                                                              c10::Allocator* allocator,
                                                              bool resizable) {
  c10::intrusive_ptr<c10::StorageImpl> custom_storage_impl;
  if (data_ptr == nullptr){
    custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, resizable);
  } else {
    custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr), allocator, resizable);
  }
  storageImpl_counter += 1;
  return custom_storage_impl;
}

// Register our dummy storageImpl create method.
void custom_storage_registry() {
  c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &make_custom_storage_impl);
}

bool custom_storageImpl_called() {
  if (storageImpl_counter > last_storageImpl_saved_value) {
    last_storageImpl_saved_value = storageImpl_counter;
    return true;
  }
  return false;
}

// basic dummy add function
at::Tensor custom_add_Tensor(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  add_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

at::Tensor custom__copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    return dst.copy_(self, false);
}

// Some set operations for the basic use case
at::Tensor& custom_set_source_Storage(at::Tensor& result, c10::Storage src) {
  int64_t new_size = static_cast<int64_t>(src.nbytes() / result.dtype().itemsize());
  c10::IntArrayRef stride = {};
  result.unsafeGetTensorImpl()->set_storage_offset(0);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : std::nullopt;
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(),
                               new_size, stride_opt,
                               /*resize_storage=*/!result.is_meta());
  return result;
}

// This macro does the heavy lifting.
// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &custom_add_Tensor);
  m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);
  m.impl("set_.source_Storage", &custom_set_source_Storage);
}

void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_fused_adamw_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("triu_indices", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

bool custom_add_called() {
  bool called = false;
  if (add_counter > last_saved_value) {
    called = true;
    last_saved_value = add_counter;
  }
  return called;
}

void fallback_with_undefined_tensor() {
  at::Tensor first = at::empty({2, 3}).to(at::DeviceType::PrivateUse1);
  at::Tensor second = at::Tensor();
  at::Tensor step = at::empty({}).fill_(2).to(at::DeviceType::PrivateUse1);
  at::Tensor grad_scale = at::empty({}).fill_(0.00001).to(at::DeviceType::PrivateUse1);
  at::Tensor found_inf = at::empty({}).fill_(1).to(at::DeviceType::PrivateUse1);
  at::TensorList tensors = {first, first};
  at::TensorList undefined_tensors = {first, second};
  at::TensorList steps = {step, step};
  return at::_fused_adamw_(tensors, tensors, tensors, tensors, undefined_tensors,
                           steps, 0.001, 0.9, 0.999, 1e-2, 1e-8, false, false,
                           grad_scale, found_inf);
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
    m.def("custom_add_called", &custom_add_called, "check if our custom add function was called");
    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
    m.def("custom_storageImpl_called", &custom_storageImpl_called, "check if our custom abs function was called");
    m.def("fallback_with_undefined_tensor", &fallback_with_undefined_tensor, "fallback_with_undefined_tensor for privateuse1");
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Allocator.h`
- `c10/core/ScalarType.h`
- `c10/core/impl/DeviceGuardImplInterface.h`
- `c10/core/impl/alloc_cpu.h`
- `c10/macros/Macros.h`
- `c10/util/ArrayRef.h`
- `torch/csrc/Device.h`
- `torch/csrc/jit/serialization/pickler.h`
- `torch/extension.h`
- `ATen/EmptyTensor.h`
- `ATen/detail/PrivateUse1HooksInterface.h`
- `ATen/native/CPUFallback.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/Resize.h`
- `ATen/native/UnaryOps.h`
- `ATen/native/cpu/Loops.h`
- `ATen/native/transformers/attention.h`
- `ATen/native/transformers/sdp_utils_cpp.h`
- `ATen/ops/view.h`
- `unordered_map`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp_extensions/open_registration_extension.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions`):

- [`cpp_frontend_extension.cpp_docs.md`](./cpp_frontend_extension.cpp_docs.md)
- [`extension.cpp_docs.md`](./extension.cpp_docs.md)
- [`identity.cpp_docs.md`](./identity.cpp_docs.md)
- [`doubler.h_docs.md`](./doubler.h_docs.md)
- [`setup.py_docs.md`](./setup.py_docs.md)
- [`rng_extension.cpp_docs.md`](./rng_extension.cpp_docs.md)
- [`cusolver_extension.cpp_docs.md`](./cusolver_extension.cpp_docs.md)
- [`cuda_dlink_extension.cpp_docs.md`](./cuda_dlink_extension.cpp_docs.md)
- [`cuda_dlink_extension_add.cu_docs.md`](./cuda_dlink_extension_add.cu_docs.md)


## Cross-References

- **File Documentation**: `open_registration_extension.cpp_docs.md`
- **Keyword Index**: `open_registration_extension.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp_extensions/open_registration_extension.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions`):

- [`cpp_frontend_extension.cpp_docs.md_docs.md`](./cpp_frontend_extension.cpp_docs.md_docs.md)
- [`cuda_dlink_extension_add.cu_docs.md_docs.md`](./cuda_dlink_extension_add.cu_docs.md_docs.md)
- [`cpp_c10d_extension.cpp_docs.md_docs.md`](./cpp_c10d_extension.cpp_docs.md_docs.md)
- [`setup.py_kw.md_docs.md`](./setup.py_kw.md_docs.md)
- [`extension.cpp_kw.md_docs.md`](./extension.cpp_kw.md_docs.md)
- [`jit_extension.cpp_docs.md_docs.md`](./jit_extension.cpp_docs.md_docs.md)
- [`cuda_dlink_extension_kernel.cu_kw.md_docs.md`](./cuda_dlink_extension_kernel.cu_kw.md_docs.md)
- [`cuda_extension_kernel2.cu_kw.md_docs.md`](./cuda_extension_kernel2.cu_kw.md_docs.md)
- [`mtia_extension.cpp_kw.md_docs.md`](./mtia_extension.cpp_kw.md_docs.md)
- [`setup.py_docs.md_docs.md`](./setup.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `open_registration_extension.cpp_docs.md_docs.md`
- **Keyword Index**: `open_registration_extension.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
