# Documentation: `c10/cuda/CUDAGraphsC10Utils.h`

## File Metadata

- **Path**: `c10/cuda/CUDAGraphsC10Utils.h`
- **Size**: 2,744 bytes (2.68 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <utility>

// CUDA Graphs utils used by c10 and aten.
// aten/cuda/CUDAGraphsUtils.cuh adds utils used by aten only.

namespace c10::cuda {

// RAII guard for "cudaStreamCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
struct C10_CUDA_API CUDAStreamCaptureModeGuard {
  CUDAStreamCaptureModeGuard(cudaStreamCaptureMode desired)
      : strictness_(desired) {
    C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&strictness_));
  }
  CUDAStreamCaptureModeGuard(const CUDAStreamCaptureModeGuard&) = delete;
  CUDAStreamCaptureModeGuard(CUDAStreamCaptureModeGuard&&) = delete;
  CUDAStreamCaptureModeGuard& operator=(const CUDAStreamCaptureModeGuard&) =
      delete;
  CUDAStreamCaptureModeGuard& operator=(CUDAStreamCaptureModeGuard&&) = delete;
  ~CUDAStreamCaptureModeGuard() {
    C10_CUDA_CHECK_WARN(cudaThreadExchangeStreamCaptureMode(&strictness_));
  }

 private:
  cudaStreamCaptureMode strictness_;
};

// Protects against enum cudaStreamCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) == 0,
    "unexpected int(cudaStreamCaptureStatusNone) value");
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive) == 1,
    "unexpected int(cudaStreamCaptureStatusActive) value");
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated) == 2,
    "unexpected int(cudaStreamCaptureStatusInvalidated) value");

enum class CaptureStatus : int {
  None = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone),
  Active = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive),
  Invalidated = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "cudaStreamCaptureStatusNone";
      break;
    case CaptureStatus::Active:
      os << "cudaStreamCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "cudaStreamCaptureStatusInvalidated";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown CUDA graph CaptureStatus", int(status));
  }
  return os;
}

// Use this version where you're sure a CUDA context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  cudaStreamCaptureStatus is_capturing{cudaStreamCaptureStatusNone};
  C10_CUDA_CHECK(
      cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}

} // namespace c10::cuda

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `C10_CUDA_API`, `CaptureStatus`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/cuda`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/cuda/CUDAStream.h`
- `iostream`
- `utility`


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
- [`CUDAException.h_docs.md`](./CUDAException.h_docs.md)


## Cross-References

- **File Documentation**: `CUDAGraphsC10Utils.h_docs.md`
- **Keyword Index**: `CUDAGraphsC10Utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
