# Documentation: `docs/torch/csrc/cuda/shared/nvtx.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/cuda/shared/nvtx.cpp_docs.md`
- **Size**: 5,436 bytes (5.31 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/cuda/shared/nvtx.cpp`

## File Metadata

- **Path**: `torch/csrc/cuda/shared/nvtx.cpp`
- **Size**: 3,326 bytes (3.25 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif

#include <cuda_runtime.h>

#ifndef ROCM_ON_WINDOWS
#if CUDART_VERSION >= 13000 || defined(TORCH_CUDA_USE_NVTX3)
#include <nvtx3/nvtx3.hpp>
#else // CUDART_VERSION >= 13000 || defined(TORCH_CUDA_USE_NVTX3)
#include <nvToolsExt.h>
#endif // CUDART_VERSION >= 13000 || defined(TORCH_CUDA_USE_NVTX3)
#else // ROCM_ON_WINDOWS
#include <c10/util/Exception.h>
#endif // ROCM_ON_WINDOWS
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

#ifndef ROCM_ON_WINDOWS
struct RangeHandle {
  nvtxRangeId_t id;
  const char* msg;
};

static void device_callback_range_end(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  nvtxRangeEnd(handle->id);
  free((void*)handle->msg);
  free((void*)handle);
}

static void device_nvtxRangeEnd(void* handle, std::intptr_t stream) {
  C10_CUDA_CHECK(cudaLaunchHostFunc(
      (cudaStream_t)stream, device_callback_range_end, handle));
}

static void device_callback_range_start(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  handle->id = nvtxRangeStartA(handle->msg);
}

static void* device_nvtxRangeStart(const char* msg, std::intptr_t stream) {
  auto handle = static_cast<RangeHandle*>(calloc(1, sizeof(RangeHandle)));
  handle->msg = strdup(msg);
  handle->id = 0;
  TORCH_CHECK(
      cudaLaunchHostFunc(
          (cudaStream_t)stream, device_callback_range_start, (void*)handle) ==
      cudaSuccess);
  return handle;
}

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#ifdef TORCH_CUDA_USE_NVTX3
  auto nvtx = m.def_submodule("_nvtx", "nvtx3 bindings");
#else
  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
#endif
  nvtx.def("rangePushA", nvtxRangePushA);
  nvtx.def("rangePop", nvtxRangePop);
  nvtx.def("rangeStartA", nvtxRangeStartA);
  nvtx.def("rangeEnd", nvtxRangeEnd);
  nvtx.def("markA", nvtxMarkA);
  nvtx.def("deviceRangeStart", device_nvtxRangeStart);
  nvtx.def("deviceRangeEnd", device_nvtxRangeEnd);
}

#else // ROCM_ON_WINDOWS

static void printUnavailableWarning() {
  TORCH_WARN_ONCE("Warning: roctracer isn't available on Windows");
}

static int rangePushA(const std::string&) {
  printUnavailableWarning();
  return 0;
}

static int rangePop() {
  printUnavailableWarning();
  return 0;
}

static int rangeStartA(const std::string&) {
  printUnavailableWarning();
  return 0;
}

static void rangeEnd(int) {
  printUnavailableWarning();
}

static void markA(const std::string&) {
  printUnavailableWarning();
}

static py::object deviceRangeStart(const std::string&, std::intptr_t) {
  printUnavailableWarning();
  return py::none(); // Return an appropriate default object
}

static void deviceRangeEnd(py::object, std::intptr_t) {
  printUnavailableWarning();
}

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto nvtx = m.def_submodule("_nvtx", "unavailable");

  nvtx.def("rangePushA", rangePushA);
  nvtx.def("rangePop", rangePop);
  nvtx.def("rangeStartA", rangeStartA);
  nvtx.def("rangeEnd", rangeEnd);
  nvtx.def("markA", markA);
  nvtx.def("deviceRangeStart", deviceRangeStart);
  nvtx.def("deviceRangeEnd", deviceRangeEnd);
}
#endif // ROCM_ON_WINDOWS

} // namespace torch::cuda::shared

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `RangeHandle`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/cuda/shared`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `wchar.h`
- `cuda_runtime.h`
- `nvtx3/nvtx3.hpp`
- `nvToolsExt.h`
- `c10/util/Exception.h`
- `c10/cuda/CUDAException.h`
- `torch/csrc/utils/pybind.h`


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
- [`cudart.cpp_docs.md`](./cudart.cpp_docs.md)


## Cross-References

- **File Documentation**: `nvtx.cpp_docs.md`
- **Keyword Index**: `nvtx.cpp_kw.md`
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

- [`cusparselt.cpp_kw.md_docs.md`](./cusparselt.cpp_kw.md_docs.md)
- [`cudart.cpp_docs.md_docs.md`](./cudart.cpp_docs.md_docs.md)
- [`cudnn.cpp_docs.md_docs.md`](./cudnn.cpp_docs.md_docs.md)
- [`nvtx.cpp_kw.md_docs.md`](./nvtx.cpp_kw.md_docs.md)
- [`cudart.cpp_kw.md_docs.md`](./cudart.cpp_kw.md_docs.md)
- [`cusparselt.cpp_docs.md_docs.md`](./cusparselt.cpp_docs.md_docs.md)
- [`cudnn.cpp_kw.md_docs.md`](./cudnn.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `nvtx.cpp_docs.md_docs.md`
- **Keyword Index**: `nvtx.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
