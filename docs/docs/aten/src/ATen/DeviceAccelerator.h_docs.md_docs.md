# Documentation: `docs/aten/src/ATen/DeviceAccelerator.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/DeviceAccelerator.h_docs.md`
- **Size**: 6,817 bytes (6.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/DeviceAccelerator.h`

## File Metadata

- **Path**: `aten/src/ATen/DeviceAccelerator.h`
- **Size**: 4,131 bytes (4.03 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/DeviceType.h>
#include <c10/macros/Macros.h>

#include <ATen/detail/MTIAHooksInterface.h>
#include <optional>

namespace at::accelerator {

// Note [Accelerator Concept]
// This file defines the top level Accelerator concept for PyTorch.
// A device is an accelerator per the definition here if:
// - It is mutually exclusive with all other accelerators
// - It performs asynchronous compute via a Stream/Event system
// - It provides a set of common APIs as defined by AcceleratorHooksInterface
//
// As of today, accelerator devices are (in no particular order):
// CUDA, MTIA, XPU, HIP, MPS, PrivateUse1

// Ensures that only one accelerator is available (at
// compile time if possible) and return it.
// When checked is true, the returned optional always has a value.
TORCH_API std::optional<c10::DeviceType> getAccelerator(bool checked = false);

// Check if the given device type is an accelerator.
TORCH_API bool isAccelerator(c10::DeviceType device_type);

// Check if the given device type is an accelerator, not the excluded ones.
template <
    typename... T,
    typename = std::enable_if_t<(std::is_same_v<T, c10::DeviceType> && ...)>>
inline bool isAcceleratorExcluded(
    c10::DeviceType device_type,
    c10::DeviceType first_excluded,
    T... rest_excluded) {
  if constexpr (sizeof...(rest_excluded) > 0) {
    return device_type != first_excluded &&
        isAcceleratorExcluded(device_type, rest_excluded...);
  } else {
    return device_type != first_excluded && isAccelerator(device_type);
  }
}

// Return the number of the device available. Note that this is *REQUIRED* to
// not raise any exception.
TORCH_API c10::DeviceIndex deviceCount();

// Set the current device index to the given device index.
TORCH_API void setDeviceIndex(c10::DeviceIndex device_index);

// Get the current device index.
TORCH_API c10::DeviceIndex getDeviceIndex();

// Set the current stream to a given stream. Note that this API doesn't change
// the current device index.
TORCH_API void setCurrentStream(c10::Stream stream);

// Get the current stream of the given device index.
TORCH_API c10::Stream getCurrentStream(c10::DeviceIndex device_index);

// Wait (by blocking the calling thread) until all the work previously enqueued
// on the given device index has been completed.
TORCH_API void synchronizeDevice(c10::DeviceIndex device_index);

// Set the current device index to the given device_index and return the
// original device index that was active before the change.
TORCH_API c10::DeviceIndex exchangeDevice(c10::DeviceIndex device_index);

// Set the current device index to the given device_index. Avoid creating a new
// context if the context for device_index is not initialized. Return the
// original device index that was active before the change.
TORCH_API c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device_index);

TORCH_API inline void emptyCache() {
  const auto device_type = getAccelerator(true).value();
  at::getDeviceAllocator(device_type)->emptyCache();
}

TORCH_API inline at::CachingDeviceAllocator::DeviceStats getDeviceStats(
    c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  return at::getDeviceAllocator(device_type)->getDeviceStats(device_index);
}

TORCH_API inline void resetAccumulatedStats(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  at::getDeviceAllocator(device_type)->resetAccumulatedStats(device_index);
}

TORCH_API inline void resetPeakStats(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  at::getDeviceAllocator(device_type)->resetPeakStats(device_index);
}

TORCH_API inline std::pair<size_t, size_t> getMemoryInfo(
    c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  return at::getDeviceAllocator(device_type)->getMemoryInfo(device_index);
}
} // namespace at::accelerator

namespace at {
// Keep BC only
using at::accelerator::getAccelerator;
using at::accelerator::isAccelerator;
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

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

- `c10/core/CachingDeviceAllocator.h`
- `c10/core/DeviceType.h`
- `c10/macros/Macros.h`
- `ATen/detail/MTIAHooksInterface.h`
- `optional`


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

- **File Documentation**: `DeviceAccelerator.h_docs.md`
- **Keyword Index**: `DeviceAccelerator.h_kw.md`
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

- **File Documentation**: `DeviceAccelerator.h_docs.md_docs.md`
- **Keyword Index**: `DeviceAccelerator.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
