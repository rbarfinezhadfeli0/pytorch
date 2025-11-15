# Documentation: `aten/src/ATen/cuda/CachingHostAllocator.h`

## File Metadata

- **Path**: `aten/src/ATen/cuda/CachingHostAllocator.h`
- **Size**: 3,032 bytes (2.96 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Deprecated.h>

namespace at::cuda {

//
// A caching allocator for CUDA host allocations (pinned memory).
//
// This provides a drop-in replacement for THCudaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
//
// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a cudaMemcpyAsync
// call between host and device, and passed the corresponding context from the
// allocation. This is currently invoked by at::native::copy_kernel_cuda.
//
C10_DEPRECATED_MESSAGE(
  "at::cuda::getCachingHostAllocator() is deprecated. Please use at::getHostAllocator(at::kCUDA) instead.")
inline TORCH_CUDA_CPP_API at::HostAllocator* getCachingHostAllocator() {
  return at::getHostAllocator(at::kCUDA);
}

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_recordEvent(...) is deprecated. Please use at::getHostAllocator(at::kCUDA)->record_event(...) instead.")
inline TORCH_CUDA_CPP_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::cuda::CUDAStream stream) {
  return getHostAllocator(at::kCUDA)->record_event(ptr, ctx, stream.unwrap());
}

// Releases cached pinned memory allocations via cudaHostFree
C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_emptyCache() is deprecated. Please use at::getHostAllocator(at::kCUDA)->empty_cache() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_emptyCache() {
  getHostAllocator(at::kCUDA)->empty_cache();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::HostAlloc(...) is deprecated. Please use at::getHostAllocator(at::kCUDA)->allocate(...) instead.")
inline TORCH_CUDA_CPP_API at::DataPtr HostAlloc(size_t size) {
  return getHostAllocator(at::kCUDA)->allocate(size);
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_getStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->get_stats() instead.")
inline TORCH_CUDA_CPP_API at::HostStats CachingHostAllocator_getStats() {
  return getHostAllocator(at::kCUDA)->get_stats();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_resetAccumulatedStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->reset_accumulated_stats() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_resetAccumulatedStats() {
  getHostAllocator(at::kCUDA)->reset_accumulated_stats();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_resetPeakStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->reset_peak_stats() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_resetPeakStats() {
  getHostAllocator(at::kCUDA)->reset_peak_stats();
}

} // namespace at::cuda

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/CachingHostAllocator.h`
- `c10/core/Allocator.h`
- `c10/cuda/CUDAStream.h`
- `c10/util/Deprecated.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`aten/src/ATen/cuda`):

- [`CublasHandlePool.cpp_docs.md`](./CublasHandlePool.cpp_docs.md)
- [`llvm_basic.cpp_docs.md`](./llvm_basic.cpp_docs.md)
- [`CUDABlas.h_docs.md`](./CUDABlas.h_docs.md)
- [`jiterator.cu_docs.md`](./jiterator.cu_docs.md)
- [`CUDAGraph.h_docs.md`](./CUDAGraph.h_docs.md)
- [`llvm_jit_strings.h_docs.md`](./llvm_jit_strings.h_docs.md)
- [`llvm_complex.cpp_docs.md`](./llvm_complex.cpp_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md`](./CUDAGeneratorImpl.cpp_docs.md)
- [`cub_definitions.cuh_docs.md`](./cub_definitions.cuh_docs.md)
- [`jiterator_impl.h_docs.md`](./jiterator_impl.h_docs.md)


## Cross-References

- **File Documentation**: `CachingHostAllocator.h_docs.md`
- **Keyword Index**: `CachingHostAllocator.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
