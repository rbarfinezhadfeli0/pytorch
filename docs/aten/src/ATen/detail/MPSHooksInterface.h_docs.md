# Documentation: `aten/src/ATen/detail/MPSHooksInterface.h`

## File Metadata

- **Path**: `aten/src/ATen/detail/MPSHooksInterface.h`
- **Size**: 3,740 bytes (3.65 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
//  Copyright © 2022 Apple Inc.

#pragma once

#include <ATen/detail/AcceleratorHooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <cstddef>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")
namespace at {

struct TORCH_API MPSHooksInterface : AcceleratorHooksInterface {
  // this fails the implementation if MPSHooks functions are called, but
  // MPS backend is not present.
  #define FAIL_MPSHOOKS_FUNC(func) \
    TORCH_CHECK(false, "Cannot execute ", func, "() without MPS backend.");

  ~MPSHooksInterface() override = default;

  // Initialize the MPS library state
  void init() const override {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual bool hasMPS() const {
    return false;
  }
  virtual bool isOnMacOSorNewer(unsigned major = 13, unsigned minor = 0) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  const Generator& getDefaultGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  Generator getNewGenerator(
      [[maybe_unused]] DeviceIndex device_index) const override {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual Allocator* getMPSDeviceAllocator() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void deviceSynchronize() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void commitStream() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void* getCommandBuffer() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void* getDispatchQueue() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void emptyCache() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual size_t getCurrentAllocatedMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual size_t getDriverAllocatedMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual size_t getRecommendedMaxMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void setMemoryFraction(double /*ratio*/) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void profilerStartTrace(const std::string& mode, bool waitUntilCompleted) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void profilerStopTrace() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual uint32_t acquireEvent(bool enable_timing) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  Device getDeviceFromPtr(void* data) const override {
    TORCH_CHECK(false, "Cannot get device of pointer on MPS without ATen_mps library. ");
  }
  virtual void releaseEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void recordEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void waitForEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void synchronizeEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual bool queryEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual double elapsedTimeOfEvents(uint32_t start_event_id, uint32_t end_event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  bool hasPrimaryContext(DeviceIndex device_index) const override {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  bool isPinnedPtr(const void* data) const override {
    return false;
  }
  Allocator* getPinnedMemoryAllocator() const override {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  #undef FAIL_MPSHOOKS_FUNC
};

struct TORCH_API MPSHooksArgs {};

TORCH_DECLARE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs);
#define REGISTER_MPS_HOOKS(clsname) \
  C10_REGISTER_CLASS(MPSHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MPSHooksInterface& getMPSHooks();

} // namespace detail
} // namespace at
C10_DIAGNOSTIC_POP()

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `at`

**Classes/Structs**: `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/detail/AcceleratorHooksInterface.h`
- `c10/core/Allocator.h`
- `c10/util/Exception.h`
- `c10/util/Registry.h`
- `cstddef`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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
- [`MTIAHooksInterface.h_docs.md`](./MTIAHooksInterface.h_docs.md)


## Cross-References

- **File Documentation**: `MPSHooksInterface.h_docs.md`
- **Keyword Index**: `MPSHooksInterface.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
