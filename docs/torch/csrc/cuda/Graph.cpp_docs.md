# Documentation: `torch/csrc/cuda/Graph.cpp`

## File Metadata

- **Path**: `torch/csrc/cuda/Graph.cpp`
- **Size**: 4,660 bytes (4.55 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer
// (csrc/Module.cpp) I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at::cuda::graph_pool_handle);

  shared_ptr_class_<::at::cuda::CUDAGraph>(torch_C_m, "_CUDAGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def(
          "capture_begin",
          [](::at::cuda::CUDAGraph& self,
             std::optional<c10::cuda::MempoolId_t> pool_opt,
             const std::string& capture_error_mode) {
            cudaStreamCaptureMode capture_mode{};
            c10::cuda::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::cuda::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = cudaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cudaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cudaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::capture_end))
      .def(
          "instantiate",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::instantiate))
      .def(
          "register_generator_state",
          [](::at::cuda::CUDAGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // We've unwrapped Python object to C++ object,
            // so we could release GIL before calling into C++
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::pool))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump),
          py::arg("debug_path"))
      .def(
          "raw_cuda_graph",
          [](::at::cuda::CUDAGraph& self) {
            cudaGraph_t graph = self.raw_cuda_graph();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of cudaGraph_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "raw_cuda_graph_exec",
          [](::at::cuda::CUDAGraph& self) {
            cudaGraphExec_t graph_exec = self.raw_cuda_graph_exec();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of cudaGraphExec_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph_exec);
          },
          py::call_guard<py::gil_scoped_release>());
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `of`, `of`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/python_headers.h`
- `pybind11/chrono.h`
- `torch/csrc/jit/python/pybind_utils.h`
- `torch/csrc/utils/pybind.h`
- `ATen/cuda/CUDAGraph.h`
- `c10/cuda/CUDAGraphsC10Utils.h`


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

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

- **File Documentation**: `Graph.cpp_docs.md`
- **Keyword Index**: `Graph.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
