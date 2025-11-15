# Documentation: `docs/torch/csrc/mtia/Module.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/mtia/Module.cpp_docs.md`
- **Size**: 8,146 bytes (7.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/mtia/Module.cpp`

## File Metadata

- **Path**: `torch/csrc/mtia/Module.cpp`
- **Size**: 6,082 bytes (5.94 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/mtia/Module.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::mtia {

struct _MTIAGraph {
  // MTIA use accelerator hooks to connect pytorch and outside.
  // We need to provide the MTIAGraph class at Python layer, but the hooks only
  // support hooking functions, not classes. Thus we store all MTIAGraph C++
  // instances in a map, and use a handle to choose the right instance.
  int64_t handle_;

  _MTIAGraph(bool keep_graph = false)
      : handle_(at::detail::getMTIAHooks().mtiagraphCreate(keep_graph)) {}

  ~_MTIAGraph() {
    at::detail::getMTIAHooks().mtiagraphDestroy(handle_);
  }

  void capture_begin(at::MempoolId_t pool) {
    at::detail::getMTIAHooks().mtiagraphCaptureBegin(handle_, pool);
  }

  void capture_end() {
    at::detail::getMTIAHooks().mtiagraphCaptureEnd(handle_);
  }

  void instantiate() {
    at::detail::getMTIAHooks().mtiagraphInstantiate(handle_);
  }

  void replay() {
    at::detail::getMTIAHooks().mtiagraphReplay(handle_);
  }

  void reset() {
    at::detail::getMTIAHooks().mtiagraphReset(handle_);
  }

  at::MempoolId_t pool() {
    return at::detail::getMTIAHooks().mtiagraphPool(handle_);
  }
};

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_mtia_init", []() {
    TORCH_INTERNAL_ASSERT(!torch::utils::is_device_in_bad_fork(at::kMTIA));
    torch::utils::register_fork_handler_for_device_init(at::kMTIA);
    at::globalContext().lazyInitDevice(c10::DeviceType::MTIA);
  });

  m.def("_mtia_isBuilt", []() {
    // Check if the MTIAHooks class has been registered with the registry.
    return at::detail::isMTIAHooksBuilt();
  });

  m.def("_mtia_isInBadFork", []() {
    return torch::utils::is_device_in_bad_fork(at::kMTIA);
  });

  m.def("_mtia_getCurrentStream", [](c10::DeviceIndex device_index) {
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getCurrentStream(device_index);
  });

  m.def("_mtia_getCurrentRawStream", [](c10::DeviceIndex device_index) {
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getCurrentRawStream(device_index);
  });

  m.def("_mtia_deviceSynchronize", []() {
    torch::utils::device_lazy_init(at::kMTIA);
    at::detail::getMTIAHooks().deviceSynchronize(
        at::detail::getMTIAHooks().getCurrentDevice());
  });

  m.def("_mtia_exchangeDevice", [](c10::DeviceIndex device_index) {
    if (device_index < 0) {
      return static_cast<c10::DeviceIndex>(-1);
    }
    return at::detail::getMTIAHooks().exchangeDevice(device_index);
  });

  m.def("_mtia_maybeExchangeDevice", [](c10::DeviceIndex device_index) {
    if (device_index < 0) {
      return static_cast<c10::DeviceIndex>(-1);
    }
    return at::detail::getMTIAHooks().maybeExchangeDevice(device_index);
  });

  m.def("_mtia_getDefaultStream", [](c10::DeviceIndex device_index) {
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getDefaultStream(device_index);
  });

  m.def(
      "_mtia_setStream",
      [](int64_t stream_id,
         c10::DeviceIndex device_index,
         int64_t device_type) {
        torch::utils::device_lazy_init(at::kMTIA);
        at::detail::getMTIAHooks().setCurrentStream(c10::Stream::unpack3(
            stream_id,
            device_index,
            static_cast<c10::DeviceType>(device_type)));
      });

  m.def("_mtia_setCurrentStream", [](const c10::Stream& stream) {
    torch::utils::device_lazy_init(at::kMTIA);
    auto device = at::detail::getMTIAHooks().getCurrentDevice();
    if (device != stream.device_index()) {
      at::detail::getMTIAHooks().setCurrentDevice(stream.device_index());
    }
    at::detail::getMTIAHooks().setCurrentStream(stream);
  });

  m.def("_mtia_memoryStats", [](c10::DeviceIndex device_index) {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().memoryStats(device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_getDeviceCapability", [](c10::DeviceIndex device_index) {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().getDeviceCapability(device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_getDeviceProperties", [](c10::DeviceIndex device_index) {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().getDeviceProperties(device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_emptyCache", []() { at::detail::getMTIAHooks().emptyCache(); });

  m.def(
      "_mtia_recordMemoryHistory",
      [](const std::optional<std::string>& enabled,
         const std::string& stacks,
         size_t max_entries) {
        at::detail::getMTIAHooks().recordMemoryHistory(
            enabled, stacks, max_entries);
      });

  m.def("_mtia_memorySnapshot", []() {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().memorySnapshot(std::nullopt);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_attachOutOfMemoryObserver", [](const py::function& observer) {
    at::detail::getMTIAHooks().attachOutOfMemoryObserver(observer.ptr());
    return;
  });

  m.def("_mtia_getDeviceCount", []() {
    return at::detail::getMTIAHooks().deviceCount();
  });

  m.def("_mtia_resetPeakMemoryStats", [](c10::DeviceIndex device_index) {
    at::detail::getMTIAHooks().resetPeakMemoryStats(device_index);
  });

  py::class_<_MTIAGraph>(m, "_MTIAGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def("capture_begin", &_MTIAGraph::capture_begin)
      .def("capture_end", &_MTIAGraph::capture_end)
      .def("instantiate", &_MTIAGraph::instantiate)
      .def("replay", &_MTIAGraph::replay)
      .def("reset", &_MTIAGraph::reset)
      .def("pool", &_MTIAGraph::pool);
}

} // namespace torch::mtia

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `_MTIAGraph`, `at`, `has`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/mtia`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `c10/core/DeviceType.h`
- `c10/core/Stream.h`
- `torch/csrc/Generator.h`
- `torch/csrc/Stream.h`
- `torch/csrc/mtia/Module.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/device_lazy_init.h`
- `torch/csrc/utils/pybind.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/mtia`):

- [`Module.h_docs.md`](./Module.h_docs.md)


## Cross-References

- **File Documentation**: `Module.cpp_docs.md`
- **Keyword Index**: `Module.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/mtia`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/mtia`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/csrc/mtia`):

- [`Module.h_docs.md_docs.md`](./Module.h_docs.md_docs.md)
- [`Module.h_kw.md_docs.md`](./Module.h_kw.md_docs.md)
- [`Module.cpp_kw.md_docs.md`](./Module.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Module.cpp_docs.md_docs.md`
- **Keyword Index**: `Module.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
