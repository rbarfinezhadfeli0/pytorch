# Documentation: `docs/torch/csrc/inductor/aoti_runtime/device_utils.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_runtime/device_utils.h_docs.md`
- **Size**: 4,729 bytes (4.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_runtime/device_utils.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_runtime/device_utils.h`
- **Size**: 2,320 bytes (2.27 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.

#ifdef USE_CUDA

// FIXME: Currently, CPU and CUDA backend are mutually exclusive.
// This is a temporary workaround. We need a better way to support
// multi devices.

#include <cuda.h>
#include <cuda_runtime_api.h>

#define AOTI_RUNTIME_CUDA_CHECK(EXPR)                      \
  do {                                                     \
    const cudaError_t code = EXPR;                         \
    const char* msg = cudaGetErrorString(code);            \
    if (code != cudaSuccess) {                             \
      throw std::runtime_error(                            \
          std::string("CUDA error: ") + std::string(msg)); \
    }                                                      \
  } while (0)

namespace torch::aot_inductor {

using DeviceStreamType = cudaStream_t;

} // namespace torch::aot_inductor

#elif defined(USE_XPU)
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sstream>
#define AOTI_RUNTIME_XPU_CHECK(EXPR)                                      \
  do {                                                                    \
    const ze_result_t status = EXPR;                                      \
    if (status != ZE_RESULT_SUCCESS) {                                    \
      std::stringstream ss;                                               \
      ss << "L0 runtime error: " << std::hex << std::uppercase << status; \
      throw std::runtime_error(ss.str());                                 \
    }                                                                     \
  } while (0)

namespace torch::aot_inductor {

using DeviceStreamType = sycl::queue*;

} // namespace torch::aot_inductor

#else

#define AOTI_RUNTIME_CPU_CHECK(EXPR)               \
  bool ok = EXPR;                                  \
  if (!ok) {                                       \
    throw std::runtime_error("CPU runtime error"); \
  }

namespace torch::aot_inductor {

using DeviceStreamType = void*;

} // namespace torch::aot_inductor

#endif // USE_CUDA

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `cuda.h`
- `cuda_runtime_api.h`
- `level_zero/ze_api.h`
- `sycl/sycl.hpp`
- `sstream`


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

Files in the same folder (`torch/csrc/inductor/aoti_runtime`):

- [`sycl_runtime_wrappers.h_docs.md`](./sycl_runtime_wrappers.h_docs.md)
- [`utils_xpu.h_docs.md`](./utils_xpu.h_docs.md)
- [`utils_cuda.h_docs.md`](./utils_cuda.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`thread_local.h_docs.md`](./thread_local.h_docs.md)
- [`scalar_to_tensor.h_docs.md`](./scalar_to_tensor.h_docs.md)
- [`model_base.h_docs.md`](./model_base.h_docs.md)
- [`mini_array_ref.h_docs.md`](./mini_array_ref.h_docs.md)
- [`model.h_docs.md`](./model.h_docs.md)


## Cross-References

- **File Documentation**: `device_utils.h_docs.md`
- **Keyword Index**: `device_utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/inductor/aoti_runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/inductor/aoti_runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/inductor/aoti_runtime`):

- [`constant_type.h_kw.md_docs.md`](./constant_type.h_kw.md_docs.md)
- [`sycl_runtime_wrappers.h_kw.md_docs.md`](./sycl_runtime_wrappers.h_kw.md_docs.md)
- [`arrayref_tensor.h_kw.md_docs.md`](./arrayref_tensor.h_kw.md_docs.md)
- [`thread_local.h_docs.md_docs.md`](./thread_local.h_docs.md_docs.md)
- [`interface.h_docs.md_docs.md`](./interface.h_docs.md_docs.md)
- [`arrayref_tensor.h_docs.md_docs.md`](./arrayref_tensor.h_docs.md_docs.md)
- [`device_utils.h_kw.md_docs.md`](./device_utils.h_kw.md_docs.md)
- [`utils_xpu.h_docs.md_docs.md`](./utils_xpu.h_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`mini_array_ref.h_docs.md_docs.md`](./mini_array_ref.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `device_utils.h_docs.md_docs.md`
- **Keyword Index**: `device_utils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
