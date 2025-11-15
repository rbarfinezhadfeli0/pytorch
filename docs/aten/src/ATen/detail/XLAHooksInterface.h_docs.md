# Documentation: `aten/src/ATen/detail/XLAHooksInterface.h`

## File Metadata

- **Path**: `aten/src/ATen/detail/XLAHooksInterface.h`
- **Size**: 2,431 bytes (2.37 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <ATen/detail/AcceleratorHooksInterface.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")

namespace at {

constexpr const char* XLA_HELP =
  "This error has occurred because you are trying "
  "to use some XLA functionality, but the XLA library has not been "
  "loaded by the dynamic linker. You must load xla libraries by `import torch_xla`";

struct TORCH_API XLAHooksInterface : AcceleratorHooksInterface {
  ~XLAHooksInterface() override = default;

  void init() const override {
    TORCH_CHECK(false, "Cannot initialize XLA without torch_xla library. ", XLA_HELP);
  }

  virtual bool hasXLA() const {
    return false;
  }

  virtual std::string showConfig() const {
    TORCH_CHECK(
        false,
        "Cannot query detailed XLA version without torch_xla library. ",
        XLA_HELP);
  }

  const Generator& getDefaultGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    TORCH_CHECK(
        false, "Cannot get default XLA generator without torch_xla library. ", XLA_HELP);
  }

  Generator getNewGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    TORCH_CHECK(false, "Cannot get XLA generator without torch_xla library. ", XLA_HELP);
  }

  virtual DeviceIndex getCurrentDevice() const override {
    TORCH_CHECK(false, "Cannot get current XLA device without torch_xla library. ", XLA_HELP);
  }

  Device getDeviceFromPtr(void* /*data*/) const override {
    TORCH_CHECK(false, "Cannot get device of pointer on XLA without torch_xla library. ", XLA_HELP);
  }

  Allocator* getPinnedMemoryAllocator() const override {
    TORCH_CHECK(false, "Cannot get XLA pinned memory allocator without torch_xla library. ", XLA_HELP);
  }

  bool isPinnedPtr(const void* data) const override {
    return false;
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    TORCH_CHECK(false, "Cannot query primary context without torch_xla library. ", XLA_HELP);
  }

};

struct TORCH_API XLAHooksArgs {};

TORCH_DECLARE_REGISTRY(XLAHooksRegistry, XLAHooksInterface, XLAHooksArgs);
#define REGISTER_XLA_HOOKS(clsname) \
  C10_REGISTER_CLASS(XLAHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const XLAHooksInterface& getXLAHooks();
} // namespace detail
} // namespace at
C10_DIAGNOSTIC_POP()

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

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

- `c10/core/Device.h`
- `c10/util/Exception.h`
- `c10/util/Registry.h`
- `ATen/detail/AcceleratorHooksInterface.h`


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

- **File Documentation**: `XLAHooksInterface.h_docs.md`
- **Keyword Index**: `XLAHooksInterface.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
