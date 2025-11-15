# Documentation: `docs/torch/csrc/lazy/backend/backend_device.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/backend/backend_device.cpp_docs.md`
- **Size**: 4,838 bytes (4.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/backend/backend_device.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/backend/backend_device.cpp`
- **Size**: 2,501 bytes (2.44 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/lazy/backend/backend_device.h>

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <optional>

namespace torch::lazy {

BackendDevice::BackendDevice()
    : type_(getBackend()->GetDefaultDeviceType()),
      ordinal_(getBackend()->GetDefaultDeviceOrdinal()) {}

BackendDevice::BackendDevice(
    std::shared_ptr<BackendDeviceType>&& type,
    int64_t ordinal)
    : type_(std::move(type)), ordinal_(ordinal) {}

int8_t BackendDevice::type() const {
  TORCH_INTERNAL_ASSERT(type_);
  return type_->type;
}

std::string BackendDevice::toString() const {
  TORCH_INTERNAL_ASSERT(type_);
  return c10::str(type_->toString(), ordinal_);
}

int BackendDevice::compare(const BackendDevice& rhs) const {
  if (type() != rhs.type()) {
    return type() < rhs.type() ? -1 : +1;
  }
  return ordinal_ < rhs.ordinal_ ? -1 : (ordinal_ > rhs.ordinal_ ? +1 : 0);
}

std::ostream& operator<<(std::ostream& os, const BackendDevice& device) {
  os << device.toString();
  return os;
}

BackendDevice atenDeviceToBackendDevice(const c10::Device& device) {
  TORCH_CHECK(device.type() == at::kLazy, device);
  int64_t ordinal = device.has_index()
      ? device.index()
      : getBackend()->GetDefaultDeviceOrdinal();
  return BackendDevice(getBackend()->GetDefaultDeviceType(), ordinal);
}

// TODO(whc) refactor this: we need to support non 1 on 1 mapping for torch/XLA.
c10::Device backendDeviceToAtenDevice(const BackendDevice& device) {
  return c10::Device(
      at::kLazy, static_cast<c10::DeviceIndex>(device.ordinal()));
}

std::optional<BackendDevice> GetBackendDevice(at::ITensorListRef tensors) {
  for (auto& tensor : tensors) {
    if (auto lt = TryGetLtcTensor(tensor)) {
      return lt->GetDevice();
    }
  }
  return std::nullopt;
}

std::optional<BackendDevice> GetBackendDevice(at::TensorList tensors) {
  return GetBackendDevice(at::ITensorListRef(tensors));
}

std::optional<BackendDevice> GetBackendDevice(const at::Tensor& tensor) {
  if (auto lt = TryGetLtcTensor(tensor)) {
    return lt->GetDevice();
  }
  return std::nullopt;
}

std::optional<BackendDevice> GetBackendDevice(
    const std::optional<c10::Device>& device) {
  if (device) {
    return atenDeviceToBackendDevice(*device);
  }
  return std::nullopt;
}

std::optional<BackendDevice> GetBackendDevice() {
  return std::nullopt;
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/backend/backend_device.h`
- `c10/core/Device.h`
- `c10/util/Exception.h`
- `c10/util/StringUtil.h`
- `torch/csrc/lazy/backend/backend_interface.h`
- `torch/csrc/lazy/core/tensor.h`
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

- [`backend_device.h_docs.md`](./backend_device.h_docs.md)
- [`backend_data.h_docs.md`](./backend_data.h_docs.md)
- [`lowering_context.h_docs.md`](./lowering_context.h_docs.md)
- [`lowering_context.cpp_docs.md`](./lowering_context.cpp_docs.md)
- [`backend_interface.cpp_docs.md`](./backend_interface.cpp_docs.md)
- [`backend_interface.h_docs.md`](./backend_interface.h_docs.md)


## Cross-References

- **File Documentation**: `backend_device.cpp_docs.md`
- **Keyword Index**: `backend_device.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/backend`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/lazy/backend`):

- [`lowering_context.h_kw.md_docs.md`](./lowering_context.h_kw.md_docs.md)
- [`backend_interface.cpp_kw.md_docs.md`](./backend_interface.cpp_kw.md_docs.md)
- [`backend_device.h_docs.md_docs.md`](./backend_device.h_docs.md_docs.md)
- [`backend_device.h_kw.md_docs.md`](./backend_device.h_kw.md_docs.md)
- [`backend_interface.cpp_docs.md_docs.md`](./backend_interface.cpp_docs.md_docs.md)
- [`lowering_context.cpp_docs.md_docs.md`](./lowering_context.cpp_docs.md_docs.md)
- [`lowering_context.cpp_kw.md_docs.md`](./lowering_context.cpp_kw.md_docs.md)
- [`backend_interface.h_kw.md_docs.md`](./backend_interface.h_kw.md_docs.md)
- [`backend_device.cpp_kw.md_docs.md`](./backend_device.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `backend_device.cpp_docs.md_docs.md`
- **Keyword Index**: `backend_device.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
