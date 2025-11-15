# Documentation: `torch/csrc/stable/device_struct.h`

## File Metadata

- **Path**: `torch/csrc/stable/device_struct.h`
- **Size**: 3,153 bytes (3.08 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <string>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

using DeviceType = torch::headeronly::DeviceType;
using DeviceIndex = torch::stable::accelerator::DeviceIndex;

// The torch::stable::Device class is an approximate copy of c10::Device.
// It has some slight modifications:
// 1. TORCH_INTERNAL_ASSERT_DEBUG_ONLY -> STD_TORCH_CHECK
// 2. Has a string constructor that uses a shim function
// 3. does not include some is_{device} variants that we can add later
//
// We chose to copy it rather than moving it to headeronly as
// 1. Device is < 8 bytes so the *Handle approach used for tensor doesn't make
// sense
// 2. c10::Device is not header-only due to its string constructor.
//
// StableIValue conversions handle conversion between c10::Device (in libtorch)
// and torch::stable::Device (in stable user extensions)

class Device {
 private:
  DeviceType type_;
  DeviceIndex index_ = -1;

  void validate() {
    STD_TORCH_CHECK(
        index_ >= -1,
        "Device index must be -1 or non-negative, got ",
        static_cast<int>(index_));
    STD_TORCH_CHECK(
        type_ != DeviceType::CPU || index_ <= 0,
        "CPU device index must be -1 or zero, got ",
        static_cast<int>(index_));
  }

 public:
  // Construct a stable::Device from a DeviceType and optional device index
  // Default index is -1 (current device)
  /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {
    validate();
  }

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  // Construct a stable::Device from a string description
  // The string must follow the schema: (cpu|cuda|...)[:<device-index>]
  // Defined in device_inl.h to avoid circular dependencies
  /* implicit */ Device(const std::string& device_string);
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

  // Copy and move constructors can be default
  Device(const Device& other) = default;
  Device(Device&& other) noexcept = default;

  // Copy and move assignment operators can be default
  Device& operator=(const Device& other) = default;
  Device& operator=(Device&& other) noexcept = default;

  // Destructor can be default
  ~Device() = default;

  bool operator==(const Device& other) const noexcept {
    return type() == other.type() && index() == other.index();
  }

  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  void set_index(DeviceIndex index) {
    index_ = index;
  }

  DeviceType type() const noexcept {
    return type_;
  }

  DeviceIndex index() const noexcept {
    return index_;
  }

  bool has_index() const noexcept {
    return index_ != -1;
  }

  bool is_cuda() const noexcept {
    return type_ == DeviceType::CUDA;
  }

  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }
};

HIDDEN_NAMESPACE_END(torch, stable)

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `is`, `Device`, `a`, `a`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/stable`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/stable/accelerator.h`
- `torch/csrc/stable/c/shim.h`
- `torch/csrc/stable/version.h`
- `torch/headeronly/core/DeviceType.h`
- `torch/headeronly/macros/Macros.h`
- `torch/headeronly/util/Exception.h`
- `torch/headeronly/util/shim_utils.h`
- `string`


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

Files in the same folder (`torch/csrc/stable`):

- [`ops.h_docs.md`](./ops.h_docs.md)
- [`accelerator.h_docs.md`](./accelerator.h_docs.md)
- [`tensor_struct.h_docs.md`](./tensor_struct.h_docs.md)
- [`tensor_inl.h_docs.md`](./tensor_inl.h_docs.md)
- [`version.h_docs.md`](./version.h_docs.md)
- [`tensor.h_docs.md`](./tensor.h_docs.md)
- [`device_inl.h_docs.md`](./device_inl.h_docs.md)
- [`library.h_docs.md`](./library.h_docs.md)
- [`stableivalue_conversions.h_docs.md`](./stableivalue_conversions.h_docs.md)


## Cross-References

- **File Documentation**: `device_struct.h_docs.md`
- **Keyword Index**: `device_struct.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
