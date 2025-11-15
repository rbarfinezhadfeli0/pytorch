# Documentation: `torch/csrc/jit/python/opaque_obj.h`

## File Metadata

- **Path**: `torch/csrc/jit/python/opaque_obj.h`
- **Size**: 2,881 bytes (2.81 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <string>
#include <utility>

#include <c10/macros/Macros.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/custom_class.h>

namespace torch::jit {
struct OpaqueObject : public CustomClassHolder {
  OpaqueObject(py::object payload) : payload_(std::move(payload)) {}

  void setPayload(py::object payload) {
    payload_ = std::move(payload);
  }

  py::object getPayload() {
    return payload_;
  }

  py::object payload_;
};

static auto register_opaque_obj_class =
    torch::class_<OpaqueObject>("aten", "OpaqueObject")
        .def(
            "__eq__",
            [](const c10::intrusive_ptr<OpaqueObject>& self,
               const c10::intrusive_ptr<OpaqueObject>& other) {
              auto self_payload = self->getPayload();
              auto other_payload = other->getPayload();

              if (!self_payload.ptr() || !other_payload.ptr()) {
                return false;
              }

              py::gil_scoped_acquire gil;
              auto res = PyObject_RichCompareBool(
                  self_payload.ptr(), other_payload.ptr(), Py_EQ);
              if (res == -1) {
                throw py::error_already_set();
              }
              return res > 0;
            })
        .def_pickle(
            [](const c10::intrusive_ptr<OpaqueObject>& self) { // __getstate__
              // Since we cannot directly return the py::object due to
              // CustomClassHolder's signature limitations, we will have to
              // serialize it directly here. We also can't return py::bytes so
              // need to encode it into a string.
              py::module_ pickle = py::module_::import("pickle");
              py::module_ base64 = py::module_::import("base64");
              py::bytes pickled_payload =
                  pickle.attr("dumps")(self->getPayload());
              py::bytes encoded_payload =
                  base64.attr("b64encode")(pickled_payload);
              return std::string(encoded_payload);
            },
            [](const std::string& state) { // __setstate__
              py::module_ pickle = py::module_::import("pickle");
              py::module_ base64 = py::module_::import("base64");
              py::bytes state_bytes(state);
              py::bytes decoded_payload = base64.attr("b64decode")(state_bytes);
              py::object restored_payload =
                  pickle.attr("loads")(decoded_payload);
              return c10::make_intrusive<OpaqueObject>(restored_payload);
            })
        .def(
            "__obj_flatten__",
            [](const c10::intrusive_ptr<OpaqueObject>& self) {
              throw std::runtime_error(
                  "Unable to implement __obj_flatten__ for opaque objects.");
            });

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `OpaqueObject`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `string`
- `utility`
- `c10/macros/Macros.h`
- `pybind11/pybind11.h`
- `pybind11/pytypes.h`
- `torch/csrc/Export.h`
- `torch/csrc/utils/pybind.h`
- `torch/custom_class.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/jit/python`):

- [`python_ir.cpp_docs.md`](./python_ir.cpp_docs.md)
- [`python_custom_class.cpp_docs.md`](./python_custom_class.cpp_docs.md)
- [`python_ivalue.h_docs.md`](./python_ivalue.h_docs.md)
- [`pybind_utils.cpp_docs.md`](./pybind_utils.cpp_docs.md)
- [`update_graph_executor_opt.h_docs.md`](./update_graph_executor_opt.h_docs.md)
- [`python_tree_views.h_docs.md`](./python_tree_views.h_docs.md)
- [`python_ir.h_docs.md`](./python_ir.h_docs.md)
- [`python_tracer.h_docs.md`](./python_tracer.h_docs.md)
- [`python_arg_flatten.h_docs.md`](./python_arg_flatten.h_docs.md)


## Cross-References

- **File Documentation**: `opaque_obj.h_docs.md`
- **Keyword Index**: `opaque_obj.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
