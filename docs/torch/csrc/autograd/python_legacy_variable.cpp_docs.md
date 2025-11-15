# Documentation: `torch/csrc/autograd/python_legacy_variable.cpp`

## File Metadata

- **Path**: `torch/csrc/autograd/python_legacy_variable.cpp`
- **Size**: 5,052 bytes (4.93 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/autograd/python_legacy_variable.h>

#include <ATen/ATen.h>
#include <fmt/format.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/tensor/python_tensor.h>

using namespace at;

namespace torch::autograd {

static PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  HANDLE_TH_ERRORS
  THPObjectPtr _data;
  PyObject* data = nullptr;
  PyObject* grad_fn = nullptr;
  char is_volatile = 0;
  char requires_grad = 0;
  const char* name = nullptr;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* accepted_args[] = {
      "data", "requires_grad", "volatile", "_grad_fn", "name", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwds,
          "|ObbOz",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(accepted_args),
          &data,
          &requires_grad,
          &is_volatile,
          &grad_fn,
          &name))
    return nullptr;

  if (grad_fn == Py_None)
    grad_fn = nullptr;

  if (is_volatile) {
    auto r = PyErr_WarnEx(
        PyExc_UserWarning,
        "volatile was removed and now has no effect. Use `with torch.no_grad():` "
        "instead.",
        1);
    if (r != 0)
      throw python_error();
  }

  TORCH_CHECK_VALUE(
      !is_volatile || !requires_grad,
      "Variable can't be volatile and require_grad at the same time!");
  if (grad_fn && !THPFunction_Check(grad_fn)) {
    TORCH_CHECK_TYPE(
        false,
        "_grad_fn has to be a Function object or None, but got ",
        Py_TYPE(grad_fn)->tp_name);
  }
  Variable var;
  if (!data || data == Py_None) {
    // For legacy serialization code, create an empty tensor. This is also used
    // by nn.Parameter() with no arguments.
    auto dispatch_key = torch::tensors::get_default_dispatch_key();
    auto scalar_type = torch::tensors::get_default_scalar_type();
    auto options = TensorOptions(scalar_type)
                       .device(dispatchKeyToDeviceType(dispatch_key))
                       .layout(dispatchKeyToLayout(dispatch_key));
    var = at::empty({0}, options);
  } else if (THPVariable_Check(data)) {
    var = THPVariable_Unpack(data).detach();
  } else {
    TORCH_CHECK_TYPE(
        false,
        "Variable data has to be a tensor, but got ",
        Py_TYPE(data)->tp_name);
  }
  // We set `tensor`'s `allow_tensor_metadata_change` to true here, because we
  // want to allow the following use case for backward compatibility:
  //
  // ```python
  // var = Variable(torch.randn(2, 3))
  // var.resize_(4, 5)
  // ```
  var.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);

  TORCH_CHECK(
      !grad_fn,
      "_grad_fn argument to legacy Variable constructor is no longer supported.  "
      "Instead, please invoke your _grad_fn to produce a variable with it as the "
      "_grad_fn.");
  var.set_requires_grad(requires_grad);

  if (name) {
    impl::set_name(var, name);
  }

  if (jit::tracer::isTracing() && data && data != Py_None &&
      THPVariable_Check(data)) {
    if (auto* v = jit::tracer::getValueTrace(THPVariable_Unpack(data))) {
      jit::tracer::setValueTrace(var, v);
    }
  }

  return THPVariable_Wrap(var);
  END_HANDLE_TH_ERRORS
}

static PyTypeObject THPLegacyVariableType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._LegacyVariableBase", /* tp_name */
    0, /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPVariable_pynew /* tp_new */
};

void init_legacy_variable(PyObject* module) {
  if (PyType_Ready(&THPLegacyVariableType) < 0) {
    throw python_error();
  }
  auto obj = (PyObject*)&THPLegacyVariableType;
  Py_INCREF(obj);
  if (PyModule_AddObject(module, "_LegacyVariableBase", obj) < 0) {
    throw python_error();
  }
}

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/autograd/python_legacy_variable.h`
- `ATen/ATen.h`
- `fmt/format.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/autograd/python_function.h`
- `torch/csrc/autograd/python_variable.h`
- `torch/csrc/jit/frontend/tracer.h`
- `torch/csrc/tensor/python_tensor.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/autograd`):

- [`graph_task.h_docs.md`](./graph_task.h_docs.md)
- [`python_function.cpp_docs.md`](./python_function.cpp_docs.md)
- [`profiler.h_docs.md`](./profiler.h_docs.md)
- [`TraceTypeManual.cpp_docs.md`](./TraceTypeManual.cpp_docs.md)
- [`python_autograd.h_docs.md`](./python_autograd.h_docs.md)
- [`variable_info.cpp_docs.md`](./variable_info.cpp_docs.md)
- [`jit_decomp_interface.h_docs.md`](./jit_decomp_interface.h_docs.md)
- [`input_buffer.cpp_docs.md`](./input_buffer.cpp_docs.md)
- [`python_variable.h_docs.md`](./python_variable.h_docs.md)
- [`python_nn_functions.h_docs.md`](./python_nn_functions.h_docs.md)


## Cross-References

- **File Documentation**: `python_legacy_variable.cpp_docs.md`
- **Keyword Index**: `python_legacy_variable.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
