# Documentation: `aten/src/ATen/detail/AcceleratorHooksInterface.h`

## File Metadata

- **Path**: `aten/src/ATen/detail/AcceleratorHooksInterface.h`
- **Size**: 2,994 bytes (2.92 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Generator.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")

namespace at {

// AcceleratorHooksInterface is a shared interface provided by all
// accelerators to allow generic code.
// This interface is hook-based as it corresponds to all the functions
// that are going to be called in a generic way from the CPU code.

struct TORCH_API AcceleratorHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~AcceleratorHooksInterface() = default;

  // Whether this backend was enabled at compilation time.
  // This function should NEVER throw.
  virtual bool isBuilt() const {
    return false;
  }

  // Whether this backend can be used at runtime, meaning it was built,
  // its runtime dependencies are available (driver) and at least one
  // supported device can be used.
  // This function should NEVER throw. This function should NOT initialize the context
  // on any device (result of hasPrimaryContext below should not change).
  // While it is acceptable for this function to poison fork, it is
  // recommended to avoid doing so whenever possible.
  virtual bool isAvailable() const {
    return false;
  }

  // Whether the device at device_index is fully initialized or not.
  virtual bool hasPrimaryContext(DeviceIndex device_index) const = 0;

  virtual void init() const {
    TORCH_CHECK(false, "Backend doesn`t support init()");
  }

  virtual DeviceIndex deviceCount() const {
    return 0;
  }

  virtual void setCurrentDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support setCurrentDevice()");
  }

  virtual DeviceIndex getCurrentDevice() const {
    TORCH_CHECK(false, "Backend doesn't support getCurrentDevice()");
    return -1;
  }

  virtual DeviceIndex exchangeDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support exchangeDevice()");
    return -1;
  }

  virtual DeviceIndex maybeExchangeDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support maybeExchangeDevice()");
    return -1;
  }

  virtual bool isPinnedPtr(const void* data) const {
    return false;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false, "Backend doesn't support getPinnedMemoryAllocator()");
    return nullptr;
  }

  virtual Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK(false, "Backend doesn't support getDeviceFromPtr()");
  }

  virtual const Generator& getDefaultGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Backend doesn`t support getDefaultGenerator()");
  }

  virtual Generator getNewGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Backend doesn`t support getNewGenerator()");
  }
};

} // namespace at

C10_DIAGNOSTIC_POP()

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 24 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Generator.h`
- `c10/core/Allocator.h`
- `c10/core/Device.h`
- `c10/core/Stream.h`


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

Files in the same folder (`aten/src/ATen/detail`):

- [`FunctionTraits.h_docs.md`](./FunctionTraits.h_docs.md)
- [`HPUHooksInterface.cpp_docs.md`](./HPUHooksInterface.cpp_docs.md)
- [`PrivateUse1HooksInterface.cpp_docs.md`](./PrivateUse1HooksInterface.cpp_docs.md)
- [`MPSHooksInterface.cpp_docs.md`](./MPSHooksInterface.cpp_docs.md)
- [`PrivateUse1HooksInterface.h_docs.md`](./PrivateUse1HooksInterface.h_docs.md)
- [`CUDAHooksInterface.cpp_docs.md`](./CUDAHooksInterface.cpp_docs.md)
- [`XPUHooksInterface.h_docs.md`](./XPUHooksInterface.h_docs.md)
- [`XPUHooksInterface.cpp_docs.md`](./XPUHooksInterface.cpp_docs.md)
- [`MPSHooksInterface.h_docs.md`](./MPSHooksInterface.h_docs.md)
- [`MTIAHooksInterface.h_docs.md`](./MTIAHooksInterface.h_docs.md)


## Cross-References

- **File Documentation**: `AcceleratorHooksInterface.h_docs.md`
- **Keyword Index**: `AcceleratorHooksInterface.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
