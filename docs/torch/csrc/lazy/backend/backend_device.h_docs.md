# Documentation: `torch/csrc/lazy/backend/backend_device.h`

## File Metadata

- **Path**: `torch/csrc/lazy/backend/backend_device.h`
- **Size**: 2,928 bytes (2.86 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <memory>
#include <ostream>
#include <string>

#include <ATen/Tensor.h>
#include <c10/macros/Export.h>
#include <c10/util/Deprecated.h>
#include <optional>

namespace c10 {
struct Device;
}

namespace torch::lazy {

// Backend should extend it and define their own supported hardware types.
struct TORCH_API BackendDeviceType {
  int8_t type{(int8_t)at::kCPU};
  // Note: previous default value was '0', which actually maps to at::kCPU, at
  // least now it is explicit, we may want to make default/undefined semantics
  // more clear though
  BackendDeviceType() : type((int8_t)at::kCPU) {}
  BackendDeviceType(int8_t type) : type(type) {}

  virtual ~BackendDeviceType() = default;
  virtual std::string toString() const {
    return "Unknown";
  }
};

class TORCH_API BackendDevice {
 public:
  // The default constructor will set both the device type and ordinal
  // to backend specific defaults.
  BackendDevice();
  BackendDevice(std::shared_ptr<BackendDeviceType>&& type, int64_t ordinal);

  int8_t type() const;
  int64_t ordinal() const {
    return ordinal_;
  }

  bool operator==(const BackendDevice& other) const {
    return compare(other) == 0;
  }
  bool operator!=(const BackendDevice& other) const {
    return compare(other) != 0;
  }
  bool operator<(const BackendDevice& rhs) const {
    return compare(rhs) < 0;
  }

  std::string toString() const;

 private:
  int compare(const BackendDevice& rhs) const;

  // Use shared_ptr instead of unique_ptr so that BackendDevice can be copied.
  std::shared_ptr<BackendDeviceType> type_;
  int64_t ordinal_;
};

TORCH_API std::ostream& operator<<(
    std::ostream& os,
    const BackendDevice& device);

// Helpers for converting a c10::Device to BackendDevice and vice versa.
TORCH_API BackendDevice atenDeviceToBackendDevice(const c10::Device& device);
TORCH_API c10::Device backendDeviceToAtenDevice(const BackendDevice& device);

// Tries to extract the backend device out of the lazy tensor. Returns nullopt
// if the input is not a lazy tensor.
TORCH_API std::optional<BackendDevice> GetBackendDevice(
    const at::ITensorListRef tensors);
TORCH_API std::optional<BackendDevice> GetBackendDevice(
    const at::TensorList tensors);
TORCH_API std::optional<BackendDevice> GetBackendDevice(
    const at::Tensor& tensor);
TORCH_API std::optional<BackendDevice> GetBackendDevice(
    const std::optional<c10::Device>& device);

// For variadic template.
TORCH_API std::optional<BackendDevice> GetBackendDevice();

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Winfinite-recursion")
template <typename T, typename... Args>
std::optional<BackendDevice> GetBackendDevice(
    const T& tensor,
    const Args&... forward_tensors) {
  auto optional_device = GetBackendDevice(tensor);
  if (optional_device) {
    return optional_device;
  }
  return GetBackendDevice(forward_tensors...);
}
C10_DIAGNOSTIC_POP()

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `c10`

**Classes/Structs**: `Device`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `memory`
- `ostream`
- `string`
- `ATen/Tensor.h`
- `c10/macros/Export.h`
- `c10/util/Deprecated.h`
- `optional`


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

Files in the same folder (`torch/csrc/lazy/backend`):

- [`backend_device.cpp_docs.md`](./backend_device.cpp_docs.md)
- [`backend_data.h_docs.md`](./backend_data.h_docs.md)
- [`lowering_context.h_docs.md`](./lowering_context.h_docs.md)
- [`lowering_context.cpp_docs.md`](./lowering_context.cpp_docs.md)
- [`backend_interface.cpp_docs.md`](./backend_interface.cpp_docs.md)
- [`backend_interface.h_docs.md`](./backend_interface.h_docs.md)


## Cross-References

- **File Documentation**: `backend_device.h_docs.md`
- **Keyword Index**: `backend_device.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
