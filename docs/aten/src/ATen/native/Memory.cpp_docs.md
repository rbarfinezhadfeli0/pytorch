# Documentation: `aten/src/ATen/native/Memory.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Memory.cpp`
- **Size**: 2,625 bytes (2.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Context.h>
#include <c10/core/Storage.h>
#include <ATen/EmptyTensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CPUFunctions.h>
#else
#include <ATen/ops/_debug_has_internal_overlap_native.h>
#include <ATen/ops/_pin_memory.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/pin_memory_native.h>
#include <ATen/ops/_pin_memory_native.h>
#include <ATen/ops/empty_cpu_dispatch.h>
#endif

namespace at::native {

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

bool is_pinned(const Tensor& self, std::optional<c10::Device> device) {
  std::optional<c10::DeviceType> opt_device_type;
  if (device.has_value()) {
    TORCH_WARN_DEPRECATION(
        "The argument 'device' of Tensor.is_pinned() ",
        "is deprecated. Please do not pass this argument.")
    opt_device_type = device.value().type();
  }
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  // Use getAcceleratorHooksInterface to make is_pinned device-agnostic
  return at::globalContext().isPinnedPtr(self.storage().data(), opt_device_type);
}

Tensor pin_memory(const Tensor& self, std::optional<c10::Device> device) {
  if (device.has_value()) {
    TORCH_WARN_DEPRECATION(
        "The argument 'device' of Tensor.pin_memory() ",
        "is deprecated. Please do not pass this argument.")
  }
  // Kind of mad that I have to do two dynamic dispatches here, pretty
  // annoying
  if (self.is_pinned(device)) {
    return self;
  }
  return at::_pin_memory(self, device);
}

Tensor _pin_memory(const Tensor& self, std::optional<c10::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  // Use getAcceleratorHooksInterface to make pin_memory device-agnostic
  auto* allocator = device.has_value()?
      at::globalContext().getPinnedMemoryAllocator(device.value().type()):
      at::globalContext().getPinnedMemoryAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
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

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/MemoryOverlap.h`
- `ATen/Context.h`
- `c10/core/Storage.h`
- `ATen/EmptyTensor.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/CPUFunctions.h`
- `ATen/ops/_debug_has_internal_overlap_native.h`
- `ATen/ops/_pin_memory.h`
- `ATen/ops/is_pinned_native.h`
- `ATen/ops/pin_memory_native.h`
- `ATen/ops/_pin_memory_native.h`
- `ATen/ops/empty_cpu_dispatch.h`


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

- **File Documentation**: `Memory.cpp_docs.md`
- **Keyword Index**: `Memory.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
