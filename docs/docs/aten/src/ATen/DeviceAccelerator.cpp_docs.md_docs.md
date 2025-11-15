# Documentation: `docs/aten/src/ATen/DeviceAccelerator.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/DeviceAccelerator.cpp_docs.md`
- **Size**: 7,164 bytes (7.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/DeviceAccelerator.cpp`

## File Metadata

- **Path**: `aten/src/ATen/DeviceAccelerator.cpp`
- **Size**: 4,568 bytes (4.46 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace at::accelerator {

std::optional<c10::DeviceType> getAccelerator(bool checked) {
  // 1. Check PrivateUse1 backends
  // We explicitly allow PrivateUse1 and another device at the same time as we
  // use this for testing. Whenever a PrivateUse1 device is registered, use it
  // first.
  // Note that this check is only for hook registration and thus is NOT initializing
  // the device or poisoning fork.
  if (is_privateuse1_backend_registered()) {
    return kPrivateUse1;
  }

  // 2. Check runtime backends
  // This state is temporary, these runtime checks should be moved to compile-time
  // once they provide the new isBuilt API and we are sure they're never in the
  // same binary as another accelerator.
#define DETECT_RUNTIME_ACCELERATOR(device_name)     \
  if (at::has##device_name()) {                     \
    return k##device_name;                          \
  }

  DETECT_RUNTIME_ACCELERATOR(MTIA)

#undef DETECT_RUNTIME_ACCELERATOR

  // 2. Check compile-time backends
  std::optional<c10::DeviceType> device_type = std::nullopt;

#define DETECT_AND_ASSIGN_ACCELERATOR_COMP(device_name) \
  if (at::detail::get##device_name##Hooks().isBuilt()) {  \
    TORCH_CHECK(                                         \
        !device_type.has_value(),                        \
        "Cannot have both " #device_name " and ",             \
        device_type.value(), ".");                       \
    device_type = k##device_name;                        \
  }

  DETECT_AND_ASSIGN_ACCELERATOR_COMP(CUDA)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(XPU)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(HIP)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(MPS)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(HPU)
  if (checked) {
    TORCH_CHECK(
        device_type, "Cannot access accelerator device when none is available.")
  }
  return device_type;

#undef DETECT_AND_ASSIGN_ACCELERATOR_COMP
}

bool isAccelerator(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCUDA:
    case at::kMTIA:
    case at::kXPU:
    case at::kHIP:
    case at::kMPS:
    case at::kHPU:
    case at::kPrivateUse1:
      return true;
    default:
      return false;
  }
}

// NOLINTBEGIN(bugprone-unchecked-optional-access)
c10::DeviceIndex deviceCount() {
  const auto device_type = getAccelerator(false);
  if (!device_type.has_value()) {
    return static_cast<c10::DeviceIndex>(0);
  }
  c10::impl::VirtualGuardImpl impl(device_type.value());
  return impl.deviceCount();
}

void setDeviceIndex(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.setDevice({device_type, device_index});
}

c10::DeviceIndex getDeviceIndex() {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getDevice().index();
}

void setCurrentStream(c10::Stream stream) {
  const auto device_type = getAccelerator(true).value();
  TORCH_CHECK(
      device_type == stream.device_type(),
      "stream's device type ",
      c10::DeviceTypeName(stream.device_type()),
      " doesn't match the current accelerator ",
      c10::DeviceTypeName(device_type));
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.exchangeStream(stream);
}

c10::Stream getCurrentStream(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getStream({device_type, device_index});
}

void synchronizeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // impl.synchronizeDevice should can be safely called from any device
  impl.synchronizeDevice(device_index);
}

c10::DeviceIndex exchangeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.exchangeDevice({device_type, device_index}).index();
}

c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // Avoid creating a new context if the context for the given device_index
  // is not initialized.
  impl.uncheckedSetDevice({device_type, device_index});
  return impl.getDevice().index();
}
// NOLINTEND(bugprone-unchecked-optional-access)

} // namespace at::accelerator

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 19 function(s).

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

- `ATen/Context.h`
- `ATen/DeviceAccelerator.h`
- `c10/core/impl/VirtualGuardImpl.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `DeviceAccelerator.cpp_docs.md`
- **Keyword Index**: `DeviceAccelerator.cpp_kw.md`
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

- **File Documentation**: `DeviceAccelerator.cpp_docs.md_docs.md`
- **Keyword Index**: `DeviceAccelerator.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
