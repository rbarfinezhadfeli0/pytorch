# Documentation: `aten/src/ATen/cuda/CUDAContextLight.h`

## File Metadata

- **Path**: `aten/src/ATen/cuda/CUDAContextLight.h`
- **Size**: 3,127 bytes (3.05 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
// Light-weight version of CUDAContext.h with fewer transitive includes

#include <cstdint>
#include <map>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>

// cublasLT was introduced in CUDA 10.1 but we enable only for 11.1 that also
// added bf16 support
#include <cublasLt.h>

#ifdef CUDART_VERSION
#include <cusolverDn.h>
#endif

#if defined(USE_CUDSS)
#include <cudss.h>
#endif

#if defined(USE_ROCM)
#include <hipsolver/hipsolver.h>
#endif

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAFunctions.h>

namespace c10 {
struct Allocator;
}

namespace at::cuda {

/*
A common CUDA interface for ATen.

This interface is distinct from CUDAHooks, which defines an interface that links
to both CPU-only and CUDA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
CUDA builds.

CUDAContext, on the other hand, should be preferred by files only included in
CUDA builds. It is intended to expose CUDA functionality in a consistent
manner.

This means there is some overlap between the CUDAContext and CUDAHooks, but
the choice of which to use is simple: use CUDAContext when in a CUDA-only file,
use CUDAHooks otherwise.

Note that CUDAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single CUDA context/state.
*/

/**
 * DEPRECATED: use device_count() instead
 */
inline int64_t getNumGPUs() {
    return c10::cuda::device_count();
}

/**
 * CUDA is available if we compiled with CUDA, and there are one or more
 * devices.  If we compiled with CUDA but there is a driver problem, etc.,
 * this function will report CUDA is not available (rather than raise an error.)
 */
inline bool is_available() {
    return c10::cuda::device_count() > 0;
}

TORCH_CUDA_CPP_API cudaDeviceProp* getCurrentDeviceProperties();

TORCH_CUDA_CPP_API int warp_size();

TORCH_CUDA_CPP_API cudaDeviceProp* getDeviceProperties(c10::DeviceIndex device);

TORCH_CUDA_CPP_API bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device);

TORCH_CUDA_CPP_API c10::Allocator* getCUDADeviceAllocator();

/* Handles */
TORCH_CUDA_CPP_API cusparseHandle_t getCurrentCUDASparseHandle();
TORCH_CUDA_CPP_API cublasHandle_t getCurrentCUDABlasHandle();
TORCH_CUDA_CPP_API cublasLtHandle_t getCurrentCUDABlasLtHandle();

TORCH_CUDA_CPP_API void clearCublasWorkspaces();
TORCH_CUDA_CPP_API std::map<std::tuple<void *, void *>, at::DataPtr>& cublas_handle_stream_to_workspace();
TORCH_CUDA_CPP_API std::map<std::tuple<void *, void *>, at::DataPtr>& cublaslt_handle_stream_to_workspace();
TORCH_CUDA_CPP_API size_t getChosenWorkspaceSize();
TORCH_CUDA_CPP_API size_t getCUDABlasLtWorkspaceSize();
TORCH_CUDA_CPP_API void* getCUDABlasLtWorkspace();

#if defined(CUDART_VERSION) || defined(USE_ROCM)
TORCH_CUDA_CPP_API cusolverDnHandle_t getCurrentCUDASolverDnHandle();
#endif

#if defined(USE_CUDSS)
TORCH_CUDA_CPP_API cudssHandle_t getCurrentCudssHandle();
#endif

} // namespace at::cuda

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`, `c10`

**Classes/Structs**: `Allocator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `map`
- `cuda_runtime_api.h`
- `cusparse.h`
- `cublas_v2.h`
- `cublasLt.h`
- `cusolverDn.h`
- `cudss.h`
- `hipsolver/hipsolver.h`
- `c10/core/Allocator.h`
- `c10/cuda/CUDAFunctions.h`


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

- **File Documentation**: `CUDAContextLight.h_docs.md`
- **Keyword Index**: `CUDAContextLight.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
