# Documentation: `c10/cuda/CUDAException.h`

## File Metadata

- **Path**: `c10/cuda/CUDAException.h`
- **Size**: 4,416 bytes (4.31 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAMiscFunctions.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <cuda.h>

// Note [CHECK macro]
// ~~~~~~~~~~~~~~~~~~
// This is a macro so that AT_ERROR can get accurate __LINE__
// and __FILE__ information.  We could split this into a short
// macro and a function implementation if we pass along __LINE__
// and __FILE__, but no one has found this worth doing.

// Used to denote errors from CUDA framework.
// This needs to be declared here instead util/Exception.h for proper conversion
// during hipify.
namespace c10 {
class C10_CUDA_API CUDAError : public c10::Error {
  using Error::Error;
};
} // namespace c10

#define C10_CUDA_CHECK(EXPR)                                        \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    c10::cuda::c10_cuda_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        true);                                                      \
  } while (0)

#define C10_CUDA_CHECK_WARN(EXPR)                              \
  do {                                                         \
    const cudaError_t __err = EXPR;                            \
    if (C10_UNLIKELY(__err != cudaSuccess)) {                  \
      [[maybe_unused]] auto error_unused = cudaGetLastError(); \
      TORCH_WARN("CUDA warning: ", cudaGetErrorString(__err)); \
    }                                                          \
  } while (0)

// Indicates that a CUDA error is handled in a non-standard way
#define C10_CUDA_ERROR_HANDLED(EXPR) EXPR

// Intentionally ignore a CUDA error
#define C10_CUDA_IGNORE_ERROR(EXPR)                                   \
  do {                                                                \
    const cudaError_t __err = EXPR;                                   \
    if (C10_UNLIKELY(__err != cudaSuccess)) {                         \
      [[maybe_unused]] cudaError_t error_unused = cudaGetLastError(); \
    }                                                                 \
  } while (0)

// Clear the last CUDA error
#define C10_CUDA_CLEAR_ERROR()                                      \
  do {                                                              \
    [[maybe_unused]] cudaError_t error_unused = cudaGetLastError(); \
  } while (0)

// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

/// Launches a CUDA kernel appending to it all the information need to handle
/// device-side assertion failures. Checks that the launch was successful.
#define TORCH_DSA_KERNEL_LAUNCH(                                      \
    kernel, blocks, threads, shared_mem, stream, ...)                 \
  do {                                                                \
    auto& launch_registry =                                           \
        c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref();     \
    kernel<<<blocks, threads, shared_mem, stream>>>(                  \
        __VA_ARGS__,                                                  \
        launch_registry.get_uvm_assertions_ptr_for_current_device(),  \
        launch_registry.insert(                                       \
            __FILE__, __FUNCTION__, __LINE__, #kernel, stream.id())); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
  } while (0)

namespace c10::cuda {

/// In the event of a CUDA failure, formats a nice error message about that
/// failure and also checks for device-side assertion failures
C10_CUDA_API void c10_cuda_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const uint32_t line_number,
    const bool include_device_assertions);

} // namespace c10::cuda

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `C10_CUDA_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/cuda`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/cuda/CUDADeviceAssertionHost.h`
- `c10/cuda/CUDAMacros.h`
- `c10/cuda/CUDAMiscFunctions.h`
- `c10/macros/Macros.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`
- `cuda.h`


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

Files in the same folder (`c10/cuda`):

- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`CUDACachingAllocator.h_docs.md`](./CUDACachingAllocator.h_docs.md)
- [`CUDAAlgorithm.h_docs.md`](./CUDAAlgorithm.h_docs.md)
- [`CUDAFunctions.h_docs.md`](./CUDAFunctions.h_docs.md)
- [`CUDAAllocatorConfig.cpp_docs.md`](./CUDAAllocatorConfig.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`CUDAMallocAsyncAllocator.cpp_docs.md`](./CUDAMallocAsyncAllocator.cpp_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`CUDACachingAllocator.cpp_docs.md`](./CUDACachingAllocator.cpp_docs.md)


## Cross-References

- **File Documentation**: `CUDAException.h_docs.md`
- **Keyword Index**: `CUDAException.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
