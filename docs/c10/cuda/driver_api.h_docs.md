# Documentation: `c10/cuda/driver_api.h`

## File Metadata

- **Path**: `c10/cuda/driver_api.h`
- **Size**: 4,213 bytes (4.11 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <cuda.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

#include <c10/util/Exception.h>

#define C10_CUDA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                     \
    CUresult __err = EXPR;                                                 \
    if (__err != CUDA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      CUresult get_error_str_err [[maybe_unused]] =                        \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      if (get_error_str_err != CUDA_SUCCESS) {                             \
        TORCH_CHECK(false, "CUDA driver error: unknown error");            \
      } else {                                                             \
        TORCH_CHECK(false, "CUDA driver error: ", err_str);                \
      }                                                                    \
    }                                                                      \
  } while (0)

// The integer in the second column specifies the requested CUDA Driver API
// version. The dynamic loader will accept a driver with a newer version, but it
// ensures that the requested symbol exists in *at least* the specified version
// or earlier.

// Keep these requested versions as low as possible to maximize compatibility
// across different driver versions.

// Why do we pin to an older version instead of using the latest?
// If a user installs a newer driver, blindly resolving the symbol may bind to a
// newer version of the function with different behavior, potentially breaking
// PyTorch.

#define C10_LIBCUDA_DRIVER_API_REQUIRED(_) \
  _(cuDeviceGetAttribute, 12000)           \
  _(cuMemAddressReserve, 12000)            \
  _(cuMemRelease, 12000)                   \
  _(cuMemMap, 12000)                       \
  _(cuMemAddressFree, 12000)               \
  _(cuMemSetAccess, 12000)                 \
  _(cuMemUnmap, 12000)                     \
  _(cuMemCreate, 12000)                    \
  _(cuMemGetAllocationGranularity, 12000)  \
  _(cuMemExportToShareableHandle, 12000)   \
  _(cuMemImportFromShareableHandle, 12000) \
  _(cuMemsetD32Async, 12000)               \
  _(cuStreamWriteValue32, 12000)           \
  _(cuGetErrorString, 12000)

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12030)
#define C10_LIBCUDA_DRIVER_API_OPTIONAL(_) \
  _(cuCtxFromGreenCtx, 12080)              \
  _(cuCtxGetCurrent, 12080)                \
  _(cuCtxPopCurrent, 12080)                \
  _(cuCtxPushCurrent, 12080)               \
  _(cuCtxSetCurrent, 12080)                \
  _(cuGreenCtxCreate, 12080)               \
  _(cuGreenCtxDestroy, 12080)              \
  _(cuDevSmResourceSplitByCount, 12080)    \
  _(cuDeviceGet, 12080)                    \
  _(cuDeviceGetDevResource, 12080)         \
  _(cuDevResourceGenerateDesc, 12080)      \
  _(cuMulticastAddDevice, 12030)           \
  _(cuMulticastBindMem, 12030)             \
  _(cuMulticastCreate, 12030)              \
  _(cuMulticastUnbind, 12030)
#else
#define C10_LIBCUDA_DRIVER_API_OPTIONAL(_)
#endif

#define C10_NVML_DRIVER_API(_)            \
  _(nvmlInit_v2)                          \
  _(nvmlDeviceGetHandleByPciBusId_v2)     \
  _(nvmlDeviceGetNvLinkRemoteDeviceType)  \
  _(nvmlDeviceGetNvLinkRemotePciInfo_v2)  \
  _(nvmlDeviceGetComputeRunningProcesses) \
  _(nvmlSystemGetCudaDriverVersion_v2)

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12040)
#define C10_NVML_DRIVER_API_OPTIONAL(_) _(nvmlDeviceGetGpuFabricInfoV)
#else
#define C10_NVML_DRIVER_API_OPTIONAL(_)
#endif

namespace c10::cuda {

struct DriverAPI {
#define CREATE_MEMBER_VERSIONED(name, version) decltype(&name) name##_;
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBCUDA_DRIVER_API_REQUIRED(CREATE_MEMBER_VERSIONED)
  C10_LIBCUDA_DRIVER_API_OPTIONAL(CREATE_MEMBER_VERSIONED)
  C10_NVML_DRIVER_API(CREATE_MEMBER)
  C10_NVML_DRIVER_API_OPTIONAL(CREATE_MEMBER)
#undef CREATE_MEMBER_VERSIONED
#undef CREATE_MEMBER

  static DriverAPI* get();
  static void* get_nvml_handle();
};

} // namespace c10::cuda

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `DriverAPI`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/cuda`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `cuda.h`
- `nvml.h`
- `c10/util/Exception.h`


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
- [`CUDAException.h_docs.md`](./CUDAException.h_docs.md)


## Cross-References

- **File Documentation**: `driver_api.h_docs.md`
- **Keyword Index**: `driver_api.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
