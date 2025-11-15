# Documentation: `aten/src/ATen/detail/MTIAHooksInterface.h`

## File Metadata

- **Path**: `aten/src/ATen/detail/MTIAHooksInterface.h`
- **Size**: 5,263 bytes (5.14 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <c10/core/Stream.h>
#include <c10/util/Registry.h>

#include <c10/core/Allocator.h>

#include <ATen/detail/AcceleratorHooksInterface.h>
#include <c10/util/python_stub.h>

#include <string>
namespace at {
class Context;
}

namespace at {
constexpr const char* MTIA_HELP =
    "The MTIA backend requires MTIA extension for PyTorch;"
    "this error has occurred because you are trying "
    "to use some MTIA's functionality without MTIA extension included.";

struct TORCH_API MTIAHooksInterface : AcceleratorHooksInterface {
// this fails the implementation if MTIAHooks functions are called, but
// MTIA backend is not present.
#define FAIL_MTIAHOOKS_FUNC(func) TORCH_CHECK(false, "Cannot execute ", func, "() without MTIA backend.");

  ~MTIAHooksInterface() override = default;

  void init() const override {
    // Avoid logging here, since MTIA needs init devices first then it will know
    // how many devices are available. Make it as no-op if mtia extension is not
    // dynamically loaded.
    return;
  }

  virtual bool hasMTIA() const {
    return false;
  }

  DeviceIndex deviceCount() const override {
    return 0;
  }

  virtual void deviceSynchronize(c10::DeviceIndex /*device_index*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual std::string showConfig() const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  bool hasPrimaryContext(DeviceIndex /*device_index*/) const override {
    return false;
  }

  void setCurrentDevice(DeviceIndex /*device*/) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  DeviceIndex getCurrentDevice() const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  DeviceIndex exchangeDevice(DeviceIndex /*device*/) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  DeviceIndex maybeExchangeDevice(DeviceIndex /*device*/) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  virtual c10::Stream getCurrentStream(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::MTIA);
  }

  virtual int64_t getCurrentRawStream(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  virtual c10::Stream getDefaultStream(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::MTIA);
  }

  virtual void setCurrentStream(const c10::Stream& /*stream*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  bool isPinnedPtr(const void* /*data*/) const override {
    return false;
  }

  Allocator* getPinnedMemoryAllocator() const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual PyObject* memoryStats(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual PyObject* getDeviceCapability(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual PyObject* getDeviceProperties(DeviceIndex device) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual void emptyCache() const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual void recordMemoryHistory(const std::optional<std::string>& /*enabled*/,
                                   const std::string& /*stacks*/,
                                   size_t /*max_entries*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual PyObject* memorySnapshot(const std::optional<std::string>& local_path) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual DeviceIndex getDeviceCount() const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return 0;
  }

  virtual void resetPeakMemoryStats(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual void attachOutOfMemoryObserver(PyObject* observer) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return;
  }

  virtual bool isAvailable() const override;

  /* MTIAGraph related APIs */
  virtual int64_t mtiagraphCreate(bool keep_graph = false) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  virtual void mtiagraphDestroy(int64_t handle) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual void mtiagraphCaptureBegin(int64_t handle, MempoolId_t pool) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual void mtiagraphCaptureEnd(int64_t handle) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual void mtiagraphInstantiate(int64_t handle) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual void mtiagraphReplay(int64_t handle) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual void mtiagraphReset(int64_t handle) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual MempoolId_t mtiagraphPool(int64_t handle) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }
};

struct TORCH_API MTIAHooksArgs {};

TORCH_DECLARE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs);
#define REGISTER_MTIA_HOOKS(clsname) C10_REGISTER_CLASS(MTIAHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MTIAHooksInterface& getMTIAHooks();
TORCH_API bool isMTIAHooksBuilt();
} // namespace detail
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 32 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `at`

**Classes/Structs**: `Context`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/CachingDeviceAllocator.h`
- `c10/core/Device.h`
- `c10/util/Exception.h`
- `c10/core/Stream.h`
- `c10/util/Registry.h`
- `c10/core/Allocator.h`
- `ATen/detail/AcceleratorHooksInterface.h`
- `c10/util/python_stub.h`
- `string`


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
- [`MPSHooksInterface.h_docs.md`](./MPSHooksInterface.h_docs.md)


## Cross-References

- **File Documentation**: `MTIAHooksInterface.h_docs.md`
- **Keyword Index**: `MTIAHooksInterface.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
