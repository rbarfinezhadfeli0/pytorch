# Documentation: `docs/torch/csrc/utils/device_lazy_init.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/device_lazy_init.h_docs.md`
- **Size**: 5,270 bytes (5.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/device_lazy_init.h`

## File Metadata

- **Path**: `torch/csrc/utils/device_lazy_init.h`
- **Size**: 2,814 bytes (2.75 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/TensorOptions.h>
#include <torch/csrc/Export.h>

// device_lazy_init() is always compiled, even for CPU-only builds.

namespace torch::utils {

/**
 * This mechanism of lazy initialization is designed for each device backend.
 * Currently, CUDA and XPU follow this design. This function `device_lazy_init`
 * MUST be called before you attempt to access any Type(CUDA or XPU) object
 * from ATen, in any way. It guarantees that the device runtime status is lazily
 * initialized when the first runtime API is requested.
 *
 * Here are some common ways that a device object may be retrieved:
 *   - You call getNonVariableType or getNonVariableTypeOpt
 *   - You call toBackend() on a Type
 *
 * It's important to do this correctly, because if you forget to add it you'll
 * get an oblique error message seems like "Cannot initialize CUDA without
 * ATen_cuda library" or "Cannot initialize XPU without ATen_xpu library" if you
 * try to use CUDA or XPU functionality from a CPU-only build, which is not good
 * UX.
 */
TORCH_PYTHON_API void device_lazy_init(at::DeviceType device_type);
TORCH_PYTHON_API void set_requires_device_init(
    at::DeviceType device_type,
    bool value);

inline bool is_device_lazy_init_supported(at::DeviceType device_type) {
  // Add more devices here to enable lazy initialization.
  return (
      device_type == at::DeviceType::CUDA ||
      device_type == at::DeviceType::XPU ||
      device_type == at::DeviceType::HPU ||
      device_type == at::DeviceType::MTIA ||
      device_type == at::DeviceType::PrivateUse1);
}

inline void maybe_initialize_device(at::Device& device) {
  if (is_device_lazy_init_supported(device.type())) {
    device_lazy_init(device.type());
  }
}

inline void maybe_initialize_device(std::optional<at::Device>& device) {
  if (!device.has_value()) {
    return;
  }
  maybe_initialize_device(device.value());
}

inline void maybe_initialize_device(const at::TensorOptions& options) {
  auto device = options.device();
  maybe_initialize_device(device);
}

inline void maybe_initialize_device(
    std::optional<at::DeviceType>& device_type) {
  if (!device_type.has_value()) {
    return;
  }
  maybe_initialize_device(device_type.value());
}

bool is_device_initialized(at::DeviceType device_type);

TORCH_PYTHON_API bool is_device_in_bad_fork(at::DeviceType device_type);

TORCH_PYTHON_API void set_device_in_bad_fork(
    at::DeviceType device_type,
    bool value);

TORCH_PYTHON_API void register_fork_handler_for_device_init(
    at::DeviceType device_type);

inline void maybe_register_fork_handler_for_device_init(
    std::optional<at::DeviceType>& device_type) {
  if (!device_type.has_value()) {
    return;
  }
  register_fork_handler_for_device_init(device_type.value());
}

} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/TensorOptions.h`
- `torch/csrc/Export.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`disable_torch_function.cpp_docs.md`](./disable_torch_function.cpp_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `device_lazy_init.h_docs.md`
- **Keyword Index**: `device_lazy_init.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/utils`):

- [`python_tuples.h_kw.md_docs.md`](./python_tuples.h_kw.md_docs.md)
- [`six.h_kw.md_docs.md`](./six.h_kw.md_docs.md)
- [`tensor_types.cpp_docs.md_docs.md`](./tensor_types.cpp_docs.md_docs.md)
- [`tensor_list.h_kw.md_docs.md`](./tensor_list.h_kw.md_docs.md)
- [`verbose.h_kw.md_docs.md`](./verbose.h_kw.md_docs.md)
- [`invalid_arguments.cpp_kw.md_docs.md`](./invalid_arguments.cpp_kw.md_docs.md)
- [`tensor_apply.h_kw.md_docs.md`](./tensor_apply.h_kw.md_docs.md)
- [`cuda_enabled.h_docs.md_docs.md`](./cuda_enabled.h_docs.md_docs.md)
- [`tensor_layouts.h_docs.md_docs.md`](./tensor_layouts.h_docs.md_docs.md)
- [`variadic.h_kw.md_docs.md`](./variadic.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `device_lazy_init.h_docs.md_docs.md`
- **Keyword Index**: `device_lazy_init.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
