# Documentation: `torch/csrc/cuda/GdsFile.cpp`

## File Metadata

- **Path**: `torch/csrc/cuda/GdsFile.cpp`
- **Size**: 4,177 bytes (4.08 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/error.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/GdsFile.h>
#include <torch/csrc/utils/pybind.h>

#if defined(USE_CUFILE)
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cufile.h>

namespace {
// To get error message for cuFileRead/Write APIs that return ssize_t (-1 for
// filesystem error and a negative CUfileOpError enum value otherwise).
template <
    class T,
    std::enable_if_t<std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::string cuGDSFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status))
                               : std::string(c10::utils::str_error(errno));
}

// To get error message for Buf/Handle registration APIs that return
// CUfileError_t
template <
    class T,
    std::enable_if_t<!std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::string cuGDSFileGetErrorString(T status) {
  std::string errStr = cuGDSFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status))
    errStr.append(".").append(
        cudaGetErrorString(static_cast<cudaError_t>(status.cu_err)));
  return errStr;
}
} // namespace

void gds_load_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Read the binary file
  ssize_t ret = cuFileRead(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileRead failed: ", cuGDSFileGetErrorString(ret));
}

void gds_save_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuGDSFileGetErrorString(ret));
}

void gds_register_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  CUfileError_t status = cuFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      "cuFileBufRegister failed: ",
      cuGDSFileGetErrorString(status));
  return;
}

void gds_deregister_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  CUfileError_t status = cuFileBufDeregister(dataPtr);
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      "cuFileBufDeregister failed: ",
      cuGDSFileGetErrorString(status));
  return;
}

int64_t gds_register_handle(int fd) {
  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle{};
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    TORCH_CHECK(
        false,
        "cuFileHandleRegister failed: ",
        cuGDSFileGetErrorString(status));
  }

  // Returning cuFileHandle_t as int64_t
  return reinterpret_cast<int64_t>(cf_handle);
}

void gds_deregister_handle(int64_t handle) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  cuFileHandleDeregister(cf_handle);
}

#endif

namespace torch::cuda::shared {

void initGdsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#if defined(USE_CUFILE)
  m.def("_gds_register_handle", &gds_register_handle);
  m.def("_gds_deregister_handle", &gds_deregister_handle);
  m.def("_gds_register_buffer", &gds_register_buffer);
  m.def("_gds_deregister_buffer", &gds_deregister_buffer);
  m.def("_gds_load_storage", &gds_load_storage);
  m.def("_gds_save_storage", &gds_save_storage);
#endif
}

} // namespace torch::cuda::shared

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`

**Classes/Structs**: `T`, `T`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/error.h`
- `pybind11/pybind11.h`
- `torch/csrc/cuda/GdsFile.h`
- `torch/csrc/utils/pybind.h`
- `c10/cuda/CUDAGuard.h`
- `cuda_runtime.h`
- `cufile.h`


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

Files in the same folder (`torch/csrc/cuda`):

- [`python_comm.cpp_docs.md`](./python_comm.cpp_docs.md)
- [`python_comm.h_docs.md`](./python_comm.h_docs.md)
- [`memory_snapshot.cpp_docs.md`](./memory_snapshot.cpp_docs.md)
- [`GdsFile.h_docs.md`](./GdsFile.h_docs.md)
- [`GreenContext.cpp_docs.md`](./GreenContext.cpp_docs.md)
- [`CUDAPluggableAllocator.cpp_docs.md`](./CUDAPluggableAllocator.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`Module.h_docs.md`](./Module.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`nccl.h_docs.md`](./nccl.h_docs.md)


## Cross-References

- **File Documentation**: `GdsFile.cpp_docs.md`
- **Keyword Index**: `GdsFile.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
