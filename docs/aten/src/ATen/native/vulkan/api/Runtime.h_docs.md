# Documentation: `aten/src/ATen/native/vulkan/api/Runtime.h`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/api/Runtime.h`
- **Size**: 2,773 bytes (2.71 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <functional>
#include <memory>
#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Adapter.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// A Vulkan Runtime initializes a Vulkan instance and decouples the concept of
// Vulkan instance initialization from initialization of, and subsequent
// interactions with,  Vulkan [physical and logical] devices as a precursor to
// multi-GPU support.  The Vulkan Runtime can be queried for available Adapters
// (i.e. physical devices) in the system which in turn can be used for creation
// of a Vulkan Context (i.e. logical devices).  All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
//

enum AdapterSelector {
  First,
};

struct RuntimeConfiguration final {
  bool enableValidationMessages;
  bool initDefaultDevice;
  AdapterSelector defaultSelector;
  uint32_t numRequestedQueues;
};

class Runtime final {
 public:
  explicit Runtime(const RuntimeConfiguration);

  // Do not allow copying. There should be only one global instance of this
  // class.
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  Runtime(Runtime&&) noexcept;
  Runtime& operator=(Runtime&&) = delete;

  ~Runtime();

  using DeviceMapping = std::pair<PhysicalDevice, int32_t>;
  using AdapterPtr = std::unique_ptr<Adapter>;

 private:
  RuntimeConfiguration config_;

  VkInstance instance_;

  std::vector<DeviceMapping> device_mappings_;
  std::vector<AdapterPtr> adapters_;
  uint32_t default_adapter_i_;

  VkDebugReportCallbackEXT debug_report_callback_;

 public:
  inline VkInstance instance() const {
    return instance_;
  }

  inline Adapter* get_adapter_p() {
    VK_CHECK_COND(
        default_adapter_i_ >= 0 && default_adapter_i_ < adapters_.size(),
        "Pytorch Vulkan Runtime: Default device adapter is not set correctly!");
    return adapters_[default_adapter_i_].get();
  }

  inline Adapter* get_adapter_p(uint32_t i) {
    VK_CHECK_COND(
        i >= 0 && i < adapters_.size(),
        "Pytorch Vulkan Runtime: Adapter at index ",
        i,
        " is not available!");
    return adapters_[i].get();
  }

  inline uint32_t default_adapter_i() const {
    return default_adapter_i_;
  }

  using Selector =
      std::function<uint32_t(const std::vector<Runtime::DeviceMapping>&)>;
  uint32_t create_adapter(const Selector&);
};

// The global runtime is retrieved using this function, where it is declared as
// a static local variable.
Runtime* runtime();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `api`, `native`, `at`

**Classes/Structs**: `RuntimeConfiguration`, `Runtime`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/api`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `functional`
- `memory`
- `ATen/native/vulkan/api/vk_api.h`
- `ATen/native/vulkan/api/Adapter.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`aten/src/ATen/native/vulkan/api`):

- [`Shader.h_docs.md`](./Shader.h_docs.md)
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`Pipeline.h_docs.md`](./Pipeline.h_docs.md)
- [`Adapter.h_docs.md`](./Adapter.h_docs.md)
- [`Adapter.cpp_docs.md`](./Adapter.cpp_docs.md)
- [`Types.h_docs.md`](./Types.h_docs.md)
- [`QueryPool.h_docs.md`](./QueryPool.h_docs.md)
- [`Allocator.h_docs.md`](./Allocator.h_docs.md)
- [`Command.cpp_docs.md`](./Command.cpp_docs.md)
- [`Descriptor.h_docs.md`](./Descriptor.h_docs.md)


## Cross-References

- **File Documentation**: `Runtime.h_docs.md`
- **Keyword Index**: `Runtime.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
