# Documentation: `docs/torch/csrc/inductor/aoti_runner/pybind.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_runner/pybind.cpp_docs.md`
- **Size**: 11,699 bytes (11.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_runner/pybind.cpp`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_runner/pybind.cpp`
- **Size**: 8,697 bytes (8.49 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#ifdef USE_XPU
#include <torch/csrc/inductor/aoti_runner/model_container_runner_xpu.h>
#endif
#ifdef __APPLE__
#include <torch/csrc/inductor/aoti_runner/model_container_runner_mps.h>
#endif
#include <torch/csrc/inductor/aoti_runner/pybind.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <torch/csrc/utils/pybind.h>

namespace torch::inductor {

void initAOTIRunnerBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_aoti");

  py::class_<AOTIModelContainerRunnerCpu>(m, "AOTIModelContainerRunnerCpu")
      .def(py::init<const std::string&, int>())
      .def(
          "run",
          &AOTIModelContainerRunnerCpu::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerCpu::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerCpu::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerCpu::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerCpu::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerCpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerCpu::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerCpu::swap_constant_buffer)
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerCpu::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerCpu::update_constant_buffer_from_blob,
          py::arg("weights_path"));

#ifdef USE_CUDA
  py::class_<AOTIModelContainerRunnerCuda>(m, "AOTIModelContainerRunnerCuda")
      .def(py::init<const std::string&, int>())
      .def(py::init<const std::string&, int, const std::string&>())
      .def(py::init<
           const std::string&,
           int,
           const std::string&,
           const std::string&>())
      .def(py::init<
           const std::string&,
           int,
           const std::string&,
           const std::string&,
           const bool>())
      .def(
          "run",
          &AOTIModelContainerRunnerCuda::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerCuda::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerCuda::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerCuda::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerCuda::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerCuda::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerCuda::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerCuda::swap_constant_buffer)
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerCuda::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerCuda::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif
#ifdef USE_XPU
  py::class_<AOTIModelContainerRunnerXpu>(m, "AOTIModelContainerRunnerXpu")
      .def(py::init<const std::string&, int>())
      .def(py::init<const std::string&, int, const std::string&>())
      .def(py::init<
           const std::string&,
           int,
           const std::string&,
           const std::string&>())
      .def(
          "run",
          &AOTIModelContainerRunnerXpu::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerXpu::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerXpu::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerXpu::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerXpu::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerXpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerXpu::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerXpu::swap_constant_buffer)
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerXpu::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerXpu::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif
#if defined(USE_MPS) && defined(__APPLE__) && \
    !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
  py::class_<AOTIModelContainerRunnerMps>(m, "AOTIModelContainerRunnerMps")
      .def(py::init<const std::string&, int>())
      .def(
          "run",
          &AOTIModelContainerRunnerMps::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerMps::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerMps::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerMps::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerMps::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerMps::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerMps::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerMps::swap_constant_buffer)
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerMps::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerMps::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif

  m.def(
      "unsafe_alloc_void_ptrs_from_tensors",
      [](const std::vector<at::Tensor>& tensors) {
        std::vector<AtenTensorHandle> handles =
            torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(tensors);
        std::vector<void*> result(
            reinterpret_cast<void**>(handles.data()),
            reinterpret_cast<void**>(handles.data()) + handles.size());
        return result;
      });
  m.def("unsafe_alloc_void_ptr_from_tensor", [](at::Tensor& tensor) {
    return reinterpret_cast<void*>(
        torch::aot_inductor::new_tensor_handle(std::move(tensor)));
  });
  m.def(
      "alloc_tensors_by_stealing_from_void_ptrs",
      [](std::vector<void*>& raw_handles) {
        return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
            reinterpret_cast<AtenTensorHandle*>(raw_handles.data()),
            raw_handles.size());
      });
  m.def("alloc_tensor_by_stealing_from_void_ptr", [](void* raw_handle) {
    return *torch::aot_inductor::tensor_handle_to_tensor_pointer(
        reinterpret_cast<AtenTensorHandle>(raw_handle));
  });
}
} // namespace torch::inductor

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_runner`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h`
- `torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h`
- `torch/csrc/inductor/aoti_runner/model_container_runner_xpu.h`
- `torch/csrc/inductor/aoti_runner/model_container_runner_mps.h`
- `torch/csrc/inductor/aoti_runner/pybind.h`
- `torch/csrc/inductor/aoti_torch/tensor_converter.h`
- `torch/csrc/inductor/aoti_torch/utils.h`
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

Files in the same folder (`torch/csrc/inductor/aoti_runner`):

- [`model_container_runner_cuda.cpp_docs.md`](./model_container_runner_cuda.cpp_docs.md)
- [`model_container_runner_xpu.cpp_docs.md`](./model_container_runner_xpu.cpp_docs.md)
- [`model_container_runner_mps.cpp_docs.md`](./model_container_runner_mps.cpp_docs.md)
- [`model_container_runner_cpu.h_docs.md`](./model_container_runner_cpu.h_docs.md)
- [`model_container_runner_cuda.h_docs.md`](./model_container_runner_cuda.h_docs.md)
- [`model_container_runner.h_docs.md`](./model_container_runner.h_docs.md)
- [`model_container_runner_xpu.h_docs.md`](./model_container_runner_xpu.h_docs.md)
- [`model_container_runner_mps.h_docs.md`](./model_container_runner_mps.h_docs.md)
- [`model_container_runner.cpp_docs.md`](./model_container_runner.cpp_docs.md)


## Cross-References

- **File Documentation**: `pybind.cpp_docs.md`
- **Keyword Index**: `pybind.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/inductor/aoti_runner`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/inductor/aoti_runner`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/inductor/aoti_runner`):

- [`model_container_runner_xpu.cpp_kw.md_docs.md`](./model_container_runner_xpu.cpp_kw.md_docs.md)
- [`model_container_runner.cpp_kw.md_docs.md`](./model_container_runner.cpp_kw.md_docs.md)
- [`model_container_runner_cuda.h_docs.md_docs.md`](./model_container_runner_cuda.h_docs.md_docs.md)
- [`model_container_runner_xpu.h_kw.md_docs.md`](./model_container_runner_xpu.h_kw.md_docs.md)
- [`model_container_runner_cuda.h_kw.md_docs.md`](./model_container_runner_cuda.h_kw.md_docs.md)
- [`model_container_runner.cpp_docs.md_docs.md`](./model_container_runner.cpp_docs.md_docs.md)
- [`model_container_runner_cpu.cpp_kw.md_docs.md`](./model_container_runner_cpu.cpp_kw.md_docs.md)
- [`model_container_runner_cuda.cpp_kw.md_docs.md`](./model_container_runner_cuda.cpp_kw.md_docs.md)
- [`model_container_runner_cuda.cpp_docs.md_docs.md`](./model_container_runner_cuda.cpp_docs.md_docs.md)
- [`pybind.h_docs.md_docs.md`](./pybind.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `pybind.cpp_docs.md_docs.md`
- **Keyword Index**: `pybind.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
