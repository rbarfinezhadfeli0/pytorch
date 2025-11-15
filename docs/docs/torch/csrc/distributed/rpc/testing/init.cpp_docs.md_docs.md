# Documentation: `docs/torch/csrc/distributed/rpc/testing/init.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/testing/init.cpp_docs.md`
- **Size**: 7,574 bytes (7.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/testing/init.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/testing/init.cpp`
- **Size**: 5,121 bytes (5.00 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/testing/testing.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/chrono.h>

#include <utility>

namespace torch::distributed::rpc::testing {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* faulty_agent_init(PyObject* _unused, PyObject* noargs) {
  // Add the FaultyTensorPipeAgent and its backend options object
  // to the python module torch._C._distributed_rpc_testing
  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m = torch_C_m.def_submodule(
      "_distributed_rpc_testing", "distributed rpc testing bindings");
  auto module = py::handle(m).cast<py::module>();

  // Import the rpc_module so we can subclass TensorPipeAgent
  py::module rpc_module = py::module::import("torch.distributed.rpc");

#ifdef USE_TENSORPIPE
  shared_ptr_class_<FaultyTensorPipeRpcBackendOptions>(
      module,
      "FaultyTensorPipeRpcBackendOptions",
      rpc_module.attr("_TensorPipeRpcBackendOptionsBase"))
      .def(
          py::init<
              int,
              float,
              std::string,
              std::vector<std::string>,
              std::unordered_map<std::string, float>,
              int>(),
          py::arg("num_worker_threads"),
          py::arg("rpc_timeout"),
          py::arg("init_method"),
          py::arg("messages_to_fail"),
          py::arg("messages_to_delay"),
          py::arg("num_fail_sends"))
      .def_readwrite(
          "num_worker_threads", &TensorPipeRpcBackendOptions::numWorkerThreads)
      .def_readwrite(
          "messages_to_fail",
          &FaultyTensorPipeRpcBackendOptions::messagesToFail)
      .def_readwrite(
          "messages_to_delay",
          &FaultyTensorPipeRpcBackendOptions::messagesToDelay)
      .def_readwrite(
          "num_fail_sends", &FaultyTensorPipeRpcBackendOptions::numFailSends);

  shared_ptr_class_<FaultyTensorPipeAgent>(
      module, "FaultyTensorPipeAgent", rpc_module.attr("TensorPipeAgent"))
      .def(
          py::init(
              [](const c10::intrusive_ptr<::c10d::Store>& store,
                 std::string name,
                 worker_id_t rank,
                 int world_size,
                 FaultyTensorPipeRpcBackendOptions opts,
                 std::unordered_map<std::string, DeviceMap> reverse_device_maps,
                 std::vector<c10::Device> devices) {
                return std::shared_ptr<FaultyTensorPipeAgent>(
                    new FaultyTensorPipeAgent(
                        store,
                        std::move(name),
                        rank,
                        world_size,
                        std::move(opts),
                        std::move(reverse_device_maps),
                        std::move(devices),
                        std::make_unique<RequestCallbackImpl>()),
                    impl::destroy_without_gil<FaultyTensorPipeAgent>);
              }),
          py::arg("store"),
          py::arg("name"),
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("opts"),
          py::arg("reverse_device_maps"),
          py::arg("devices"))
      .def(
          "join",
          &TensorPipeAgent::join,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("shutdown") = false,
          py::arg("timeout") = 0)
      .def(
          "shutdown",
          &TensorPipeAgent::shutdown,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          static_cast<const WorkerInfo& (TensorPipeAgent::*)(void) const>(
              &RpcAgent::getWorkerInfo),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          static_cast<const WorkerInfo& (TensorPipeAgent::*)(const std::string&)
                          const>(&TensorPipeAgent::getWorkerInfo),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          static_cast<const WorkerInfo& (TensorPipeAgent::*)(worker_id_t id)
                          const>(&TensorPipeAgent::getWorkerInfo),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_infos",
          static_cast<std::vector<WorkerInfo> (TensorPipeAgent::*)() const>(
              &TensorPipeAgent::getWorkerInfos),
          py::call_guard<py::gil_scoped_release>());
#endif // USE_TENSORPIPE

  Py_RETURN_TRUE;
}

} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_faulty_agent_init", faulty_agent_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace torch::distributed::rpc::testing

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `static`, `torch`

**Classes/Structs**: `TensorPipeAgent`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc/testing`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/python_headers.h`
- `torch/csrc/distributed/rpc/request_callback_impl.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/distributed/rpc/tensorpipe_agent.h`
- `torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h`
- `torch/csrc/distributed/rpc/testing/testing.h`
- `torch/csrc/utils/pybind.h`
- `pybind11/chrono.h`
- `utility`


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

This is a test file. Run it with:

```bash
python torch/csrc/distributed/rpc/testing/init.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/distributed/rpc/testing`):

- [`testing.h_docs.md`](./testing.h_docs.md)
- [`faulty_tensorpipe_agent.cpp_docs.md`](./faulty_tensorpipe_agent.cpp_docs.md)
- [`faulty_tensorpipe_agent.h_docs.md`](./faulty_tensorpipe_agent.h_docs.md)


## Cross-References

- **File Documentation**: `init.cpp_docs.md`
- **Keyword Index**: `init.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/rpc/testing`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/rpc/testing`, which is part of the **core PyTorch library**.



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

This is a test file. Run it with:

```bash
python docs/torch/csrc/distributed/rpc/testing/init.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/distributed/rpc/testing`):

- [`faulty_tensorpipe_agent.h_kw.md_docs.md`](./faulty_tensorpipe_agent.h_kw.md_docs.md)
- [`init.cpp_kw.md_docs.md`](./init.cpp_kw.md_docs.md)
- [`faulty_tensorpipe_agent.cpp_kw.md_docs.md`](./faulty_tensorpipe_agent.cpp_kw.md_docs.md)
- [`faulty_tensorpipe_agent.cpp_docs.md_docs.md`](./faulty_tensorpipe_agent.cpp_docs.md_docs.md)
- [`testing.h_kw.md_docs.md`](./testing.h_kw.md_docs.md)
- [`testing.h_docs.md_docs.md`](./testing.h_docs.md_docs.md)
- [`faulty_tensorpipe_agent.h_docs.md_docs.md`](./faulty_tensorpipe_agent.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `init.cpp_docs.md_docs.md`
- **Keyword Index**: `init.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
