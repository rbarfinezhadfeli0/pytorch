# Documentation: driver_api.h

## File Metadata
- **Path**: `c10/cuda/driver_api.h`
- **Size**: 4213 bytes
- **Lines**: 103
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
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

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Structures
This file defines 1 struct(s): DriverAPI


## Key Components

The file contains 336 words across 103 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4213 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
