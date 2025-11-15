# Documentation: `docs/aten/src/ATen/xpu/detail/XPUHooks.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/xpu/detail/XPUHooks.cpp_docs.md`
- **Size**: 5,421 bytes (5.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/xpu/detail/XPUHooks.cpp`

## File Metadata

- **Path**: `aten/src/ATen/xpu/detail/XPUHooks.cpp`
- **Size**: 3,201 bytes (3.13 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/xpu/PeerToPeerAccess.h>
#include <ATen/xpu/PinnedMemoryAllocator.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUDevice.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <ATen/xpu/detail/XPUHooks.h>
#include <c10/util/Logging.h>
#include <c10/xpu/XPUCachingAllocator.h>

namespace at::xpu::detail {

void XPUHooks::init() const {
  C10_LOG_API_USAGE_ONCE("aten.init.xpu");
  const auto device_count = c10::xpu::device_count_ensure_non_zero();
  c10::xpu::XPUCachingAllocator::init(device_count);
  at::xpu::detail::init_p2p_access_cache(device_count);
}

bool XPUHooks::hasXPU() const {
  return true;
}

std::string XPUHooks::showConfig() const {
  return "XPU backend";
}

int32_t XPUHooks::getGlobalIdxFromDevice(const at::Device& device) const {
  TORCH_CHECK(device.is_xpu(), "Only the XPU device type is expected.");
#if defined(_WIN32) && SYCL_COMPILER_VERSION < 20250000
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "Default context is not supported on XPU by default on Windows for SYCL compiler versions earlier than 2025.0.0. So we can NOT find its global index of the ATen device.");
#else
  return at::xpu::getGlobalIdxFromDevice(device.index());
#endif
}

const Generator& XPUHooks::getDefaultGenerator(DeviceIndex device_index) const {
  return at::xpu::detail::getDefaultXPUGenerator(device_index);
}

Generator XPUHooks::getNewGenerator(DeviceIndex device_index) const {
  return make_generator<at::XPUGeneratorImpl>(device_index);
}

Device XPUHooks::getDeviceFromPtr(void* data) const {
#if defined(_WIN32) && SYCL_COMPILER_VERSION < 20250000
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "Default context is not supported on XPU by default on Windows for SYCL compiler versions earlier than 2025.0.0. So we can NOT find the ATen device of a pointer.");
#else
  return at::xpu::getDeviceFromPtr(data);
#endif
}

/**
 * DEPRECATED: use deviceCount() instead
 */
c10::DeviceIndex XPUHooks::getNumGPUs() const {
  return at::xpu::device_count();
}

/**
 * DEPRECATED: use getCurrentDevice() instead
 */
DeviceIndex XPUHooks::current_device() const {
  return c10::xpu::current_device();
}

void XPUHooks::deviceSynchronize(DeviceIndex device_index) const {
  // Only the SYCL queues we have reserved will be synchronized, see Note
  // [Synchronize Streams on Device].
  c10::xpu::syncStreamsOnDevice(device_index);
}

Allocator* XPUHooks::getPinnedMemoryAllocator() const {
  return at::xpu::getPinnedMemoryAllocator();
}

bool XPUHooks::isPinnedPtr(const void* data) const {
  if (!at::xpu::is_available()) {
    return false;
  }

  return sycl::usm::alloc::host ==
      sycl::get_pointer_type(data, c10::xpu::get_device_context());
}

bool XPUHooks::isAvailable() const {
  return at::xpu::is_available();
}

bool XPUHooks::hasPrimaryContext(DeviceIndex device_index) const {
  // The default context is utilized for each device.
  // So it always returns true if a device is available.
  return isAvailable();
}

DeviceIndex XPUHooks::deviceCount() const {
  return at::xpu::device_count();
}

DeviceIndex XPUHooks::getCurrentDevice() const {
  return at::xpu::current_device();
}

REGISTER_XPU_HOOKS(XPUHooks);

} // namespace at::xpu::detail

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/xpu/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/xpu/PeerToPeerAccess.h`
- `ATen/xpu/PinnedMemoryAllocator.h`
- `ATen/xpu/XPUContext.h`
- `ATen/xpu/XPUDevice.h`
- `ATen/xpu/XPUGeneratorImpl.h`
- `ATen/xpu/detail/XPUHooks.h`
- `c10/util/Logging.h`
- `c10/xpu/XPUCachingAllocator.h`


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

Files in the same folder (`aten/src/ATen/xpu/detail`):

- [`XPUHooks.h_docs.md`](./XPUHooks.h_docs.md)


## Cross-References

- **File Documentation**: `XPUHooks.cpp_docs.md`
- **Keyword Index**: `XPUHooks.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/xpu/detail`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/xpu/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/xpu/detail`):

- [`XPUHooks.h_kw.md_docs.md`](./XPUHooks.h_kw.md_docs.md)
- [`XPUHooks.cpp_kw.md_docs.md`](./XPUHooks.cpp_kw.md_docs.md)
- [`XPUHooks.h_docs.md_docs.md`](./XPUHooks.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `XPUHooks.cpp_docs.md_docs.md`
- **Keyword Index**: `XPUHooks.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
