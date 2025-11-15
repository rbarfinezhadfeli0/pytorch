# Documentation: `docs/torch/csrc/cuda/shared/cudart.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/cuda/shared/cudart.cpp_docs.md`
- **Size**: 5,500 bytes (5.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/cuda/shared/cudart.cpp`

## File Metadata

- **Path**: `torch/csrc/cuda/shared/cudart.cpp`
- **Size**: 3,411 bytes (3.33 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#if !defined(USE_ROCM)
#include <cuda_profiler_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace torch::cuda::shared {

#ifdef USE_ROCM
namespace {
hipError_t hipReturnSuccess() {
  return hipSuccess;
}
} // namespace
#endif

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  // cudaOutputMode_t is used in cudaProfilerInitialize only. The latter is gone
  // in CUDA 12.
  py::enum_<cudaOutputMode_t>(
      cudart,
      "cuda"
      "OutputMode")
      .value("KeyValuePair", cudaKeyValuePair)
      .value("CSV", cudaCSV);
#endif

  py::enum_<cudaError_t>(
      cudart,
      "cuda"
      "Error")
      .value("success", cudaSuccess);

  cudart.def(
      "cuda"
      "GetErrorString",
      cudaGetErrorString);
  cudart.def(
      "cuda"
      "ProfilerStart",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      cudaProfilerStart
#endif
  );
  cudart.def(
      "cuda"
      "ProfilerStop",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      cudaProfilerStop
#endif
  );
  cudart.def(
      "cuda"
      "HostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> cudaError_t {
        py::gil_scoped_release no_gil;
        return C10_CUDA_ERROR_HANDLED(
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            cudaHostRegister((void*)ptr, size, flags));
      });
  cudart.def(
      "cuda"
      "HostUnregister",
      [](uintptr_t ptr) -> cudaError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_CUDA_ERROR_HANDLED(cudaHostUnregister((void*)ptr));
      });
  cudart.def(
      "cuda"
      "StreamCreate",
      [](uintptr_t ptr) -> cudaError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_CUDA_ERROR_HANDLED(cudaStreamCreate((cudaStream_t*)ptr));
      });
  cudart.def(
      "cuda"
      "StreamDestroy",
      [](uintptr_t ptr) -> cudaError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_CUDA_ERROR_HANDLED(cudaStreamDestroy((cudaStream_t)ptr));
      });
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  // cudaProfilerInitialize is no longer needed after CUDA 12:
  // https://forums.developer.nvidia.com/t/cudaprofilerinitialize-is-deprecated-alternative/200776/3
  cudart.def(
      "cuda"
      "ProfilerInitialize",
      cudaProfilerInitialize,
      py::call_guard<py::gil_scoped_release>());
#endif
  cudart.def(
      "cuda"
      "MemGetInfo",
      [](c10::DeviceIndex device) -> std::pair<size_t, size_t> {
        c10::cuda::CUDAGuard guard(device);
        size_t device_free = 0;
        size_t device_total = 0;
        py::gil_scoped_release no_gil;
        C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
        return {device_free, device_total};
      });
}

} // namespace torch::cuda::shared

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/cuda/shared`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `cuda.h`
- `cuda_runtime.h`
- `torch/csrc/utils/pybind.h`
- `cuda_profiler_api.h`
- `hip/hip_runtime_api.h`
- `c10/cuda/CUDAException.h`
- `c10/cuda/CUDAGuard.h`


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

Files in the same folder (`torch/csrc/cuda/shared`):

- [`cudnn.cpp_docs.md`](./cudnn.cpp_docs.md)
- [`cusparselt.cpp_docs.md`](./cusparselt.cpp_docs.md)
- [`nvtx.cpp_docs.md`](./nvtx.cpp_docs.md)


## Cross-References

- **File Documentation**: `cudart.cpp_docs.md`
- **Keyword Index**: `cudart.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/cuda/shared`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/cuda/shared`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/cuda/shared`):

- [`nvtx.cpp_docs.md_docs.md`](./nvtx.cpp_docs.md_docs.md)
- [`cusparselt.cpp_kw.md_docs.md`](./cusparselt.cpp_kw.md_docs.md)
- [`cudnn.cpp_docs.md_docs.md`](./cudnn.cpp_docs.md_docs.md)
- [`nvtx.cpp_kw.md_docs.md`](./nvtx.cpp_kw.md_docs.md)
- [`cudart.cpp_kw.md_docs.md`](./cudart.cpp_kw.md_docs.md)
- [`cusparselt.cpp_docs.md_docs.md`](./cusparselt.cpp_docs.md_docs.md)
- [`cudnn.cpp_kw.md_docs.md`](./cudnn.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cudart.cpp_docs.md_docs.md`
- **Keyword Index**: `cudart.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
