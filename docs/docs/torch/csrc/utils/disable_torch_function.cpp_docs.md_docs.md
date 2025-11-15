# Documentation: `docs/torch/csrc/utils/disable_torch_function.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/disable_torch_function.cpp_docs.md`
- **Size**: 15,250 bytes (14.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/disable_torch_function.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/disable_torch_function.cpp`
- **Size**: 12,793 bytes (12.49 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <fmt/format.h>

namespace torch {
static PyObject* disabled_torch_function = nullptr;
static PyObject* disabled_torch_dispatch = nullptr;

bool torch_function_enabled() {
  return at::impl::PythonTorchFunctionTLS::get_disabled_state() ==
      at::impl::TorchFunctionDisabledState::ENABLED;
}

PyObject* disabled_torch_function_impl() {
  return disabled_torch_function;
}

void set_disabled_torch_function_impl(PyObject* value) {
  disabled_torch_function = value;
}

PyObject* disabled_torch_dispatch_impl() {
  return disabled_torch_dispatch;
}

void set_disabled_torch_dispatch_impl(PyObject* value) {
  disabled_torch_dispatch = value;
}
} // namespace torch

typedef struct {
  PyObject_HEAD
  /* Type-specific fields go here. */
  at::impl::TorchFunctionDisabledState old_state;
} DisableTorchFunctionSubclass;

static PyObject* DisableTorchFunctionSubclass__enter(
    PyObject* self,
    PyObject* unused) {
  const auto old_state = at::impl::PythonTorchFunctionTLS::get_disabled_state();
  ((DisableTorchFunctionSubclass*)self)->old_state = old_state;
  if (old_state == at::impl::TorchFunctionDisabledState::ENABLED) {
    at::impl::PythonTorchFunctionTLS::set_disabled_state(
        at::impl::TorchFunctionDisabledState::SUBCLASSES_DISABLED);
  }
  Py_RETURN_NONE;
}

static PyObject* DisableTorchFunctionSubclass__exit(
    PyObject* self,
    PyObject* unused) {
  at::impl::PythonTorchFunctionTLS::set_disabled_state(
      ((DisableTorchFunctionSubclass*)self)->old_state);
  Py_RETURN_NONE;
}

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject* unused) {
  if (torch::torch_function_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject* THPModule_isAllDisabledTorchFunction(
    PyObject* self,
    PyObject* unused) {
  if (at::impl::torch_function_all_disabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyMethodDef DisableTorchFunctionSubclass_methods[] = { // NOLINT
    {"__enter__", DisableTorchFunctionSubclass__enter, METH_NOARGS, nullptr},
    {"__exit__", DisableTorchFunctionSubclass__exit, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static PyTypeObject DisableTorchFunctionSubclassType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C.DisableTorchFunctionSubclass", /* tp_name */
    sizeof(DisableTorchFunctionSubclass), /* tp_basicsize */
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
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    DisableTorchFunctionSubclass_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    PyType_GenericAlloc, /* tp_alloc */
    PyType_GenericNew, /* tp_new */
};

PyObject* THPModule_DisableTorchFunctionSubclassType() {
  if (PyType_Ready(&DisableTorchFunctionSubclassType) < 0) {
    return nullptr;
  }

  return (PyObject*)(&DisableTorchFunctionSubclassType);
}

typedef struct {
  PyObject_HEAD
  /* Type-specific fields go here. */
  at::impl::TorchFunctionDisabledState old_state;
} DisableTorchFunction;

static PyObject* DisableTorchFunction__enter(PyObject* self, PyObject* unused) {
  ((DisableTorchFunctionSubclass*)self)->old_state =
      at::impl::PythonTorchFunctionTLS::get_disabled_state();
  at::impl::PythonTorchFunctionTLS::set_disabled_state(
      at::impl::TorchFunctionDisabledState::ALL_DISABLED);
  Py_RETURN_NONE;
}

static PyObject* DisableTorchFunction__exit(PyObject* self, PyObject* unused) {
  at::impl::PythonTorchFunctionTLS::set_disabled_state(
      ((DisableTorchFunctionSubclass*)self)->old_state);
  Py_RETURN_NONE;
}

static PyMethodDef DisableTorchFunction_methods[] = { // NOLINT
    {"__enter__", DisableTorchFunction__enter, METH_NOARGS, nullptr},
    {"__exit__", DisableTorchFunction__exit, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static PyTypeObject DisableTorchFunctionType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C.DisableTorchFunction", /* tp_name */
    sizeof(DisableTorchFunction), /* tp_basicsize */
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
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    DisableTorchFunction_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    PyType_GenericAlloc, /* tp_alloc */
    PyType_GenericNew, /* tp_new */
};

PyObject* THPModule_DisableTorchFunctionType() {
  if (PyType_Ready(&DisableTorchFunctionType) < 0) {
    return nullptr;
  }

  return (PyObject*)(&DisableTorchFunctionType);
}

PyObject* THPModule_disable_torch_function(PyObject* self, PyObject* a) {
  HANDLE_TH_ERRORS
  PyObject *func = nullptr, *types = nullptr, *args = nullptr,
           *kwargs = nullptr;
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  if (args == nullptr) {
    py_args = py::make_tuple();
  } else if (PyList_Check(args)) {
    py_args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(args));
  } else if (PyTuple_Check(args)) {
    py_args = py::reinterpret_borrow<py::tuple>(args);
  } else {
    TORCH_CHECK_TYPE(
        false,
        fmt::format("expected List or Tuple (got {})", Py_TYPE(args)->tp_name));
  }

  // These are all C-API calls so no exceptions will be raised
  // and therefore no need for RAII approach to storing
  // the old value.
  auto old_value = at::impl::PythonTorchFunctionTLS::get_disabled_state();
  if (old_value == at::impl::TorchFunctionDisabledState::ENABLED) {
    at::impl::PythonTorchFunctionTLS::set_disabled_state(
        at::impl::TorchFunctionDisabledState::SUBCLASSES_DISABLED);
  }
  // kwargs can safely be nullptr here.
  PyObject* result = PyObject_Call(func, py_args.ptr(), kwargs);
  at::impl::PythonTorchFunctionTLS::set_disabled_state(old_value);
  return result;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_disable_torch_dispatch(PyObject* self, PyObject* a) {
  HANDLE_TH_ERRORS
  PyObject *func = nullptr, *types = nullptr, *args = nullptr,
           *kwargs = nullptr;
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  if (args == nullptr) {
    py_args = py::make_tuple();
  } else if (PyList_Check(args)) {
    py_args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(args));
  } else if (PyTuple_Check(args)) {
    py_args = py::reinterpret_borrow<py::tuple>(args);
  } else {
    TORCH_CHECK_TYPE(
        false,
        fmt::format("expected List or Tuple (got {})", Py_TYPE(args)->tp_name));
  }

  // This implementation is not completely correct.  The moral
  // meaning of this function is that we should do a redispatch
  // "after" PythonKey, aka a redispatch() call.  But we don't have a
  // dispatcher call here; we have an opaque Python object.
  //
  // What we have here is a close approximation: instead of redispatch(), we
  // just exclude Python and all the keys before it, so that we will go
  // to the next key after Python.  The difference, however, is we are
  // now PERMANENTLY after Python.  We don't think there are any legitimate
  // cases where we want to go for another round on the entire dispatcher key
  // set, but if there are, then we will have to do something else here.
  c10::impl::ExcludeDispatchKeyGuard guard_(
      // TODO: add constructor for this specifically
      c10::DispatchKeySet(c10::DispatchKeySet::FULL) -
      c10::DispatchKeySet(
          c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Python)
      // NB: off by one hazard here, but it works out: python key is not
      // included in AFTER, so it is included in the negation (and that's
      // correct: we want to exclude Python key and everything BEFORE it.)
  );
  auto r = PyObject_Call(func, py_args.ptr(), kwargs);
  if (r == nullptr)
    throw python_error();
  return r;
  END_HANDLE_TH_ERRORS
}

// Makes sure that we don't check for __torch_function__ on basic Python types
static bool is_basic_python_type(PyTypeObject* tp) {
  return (
      /* Basic number types */
      tp == &PyBool_Type ||

      tp == &PyLong_Type || tp == &PyFloat_Type || tp == &PyComplex_Type ||

      /* Basic sequence types */
      tp == &PyList_Type || tp == &PyTuple_Type || tp == &PyDict_Type ||
      tp == &PySet_Type || tp == &PyFrozenSet_Type || tp == &PyUnicode_Type ||
      tp == &PyBytes_Type ||

      /* other builtins */
      tp == &PySlice_Type || tp == Py_TYPE(Py_None) ||
      tp == Py_TYPE(Py_Ellipsis) || tp == Py_TYPE(Py_NotImplemented) ||

      PyModule_Check(tp) ||
      /* sentinel to swallow trailing || */
      false);
}

inline static bool has_torch_function_attr(PyObject* obj) {
  auto attr = PyObject_FastGetAttrString(obj, "__torch_function__");
  return (
      attr.ptr() != nullptr && attr.ptr() != torch::disabled_torch_function);
}

namespace torch {
auto check_has_torch_function(PyObject* obj, bool ignore_mode) -> bool {
  if (!ignore_mode && at::impl::torch_function_mode_enabled())
    return true;
  PyTypeObject* tp = Py_TYPE(obj);
  return (
      !THPVariable_CheckTypeExact(tp) && !is_basic_python_type(tp) &&
      torch::torch_function_enabled() && has_torch_function_attr(obj));
}
} // namespace torch

inline static bool sequence_has_torch_function(PyObject* args) {
  Py_ssize_t nargs = PySequence_Fast_GET_SIZE(args);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(args, i);
    if (torch::check_has_torch_function(obj)) {
      return true;
    }
  }
  return false;
}

inline static bool array_has_torch_function(
    PyObject* const* args,
    Py_ssize_t nargs) {
  for (Py_ssize_t i = 0; i < nargs; i++) {
    if (torch::check_has_torch_function(args[i])) {
      return true;
    }
  }
  return false;
}

PyObject* THPModule_has_torch_function(PyObject* /*unused*/, PyObject* arg) {
  bool result = false;
  if (PyTuple_CheckExact(arg) || PyList_CheckExact(arg)) {
    // Fast path:
    //   If we know that we have a tuple or list, we can skip an INCREF and
    //   DECREF from PySequence_Fast. Core functions will always follow this
    //   convention (almost always tuples), and it shaves ~3.5% off the cost of
    //   the check.
    result = sequence_has_torch_function(arg);
  } else {
    auto args = py::reinterpret_steal<py::object>(
        PySequence_Fast(arg, "expected a sequence"));
    if (!args) {
      return nullptr;
    }
    result = sequence_has_torch_function(args.ptr());
  }

  if (result) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_has_torch_function_unary(
    PyObject* /*unused*/,
    PyObject* obj) {
  // Special case `THPModule_has_torch_function` for the single arg case.
  if (torch::check_has_torch_function(obj)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_has_torch_function_variadic(
    PyObject* /*unused*/,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (array_has_torch_function(args, nargs)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Exceptions.h`
- `torch/csrc/autograd/python_variable.h`
- `torch/csrc/utils/disable_torch_function.h`
- `torch/csrc/utils/pybind.h`
- `torch/csrc/utils/python_strings.h`
- `ATen/PythonTorchFunctionTLS.h`
- `fmt/format.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `disable_torch_function.cpp_docs.md`
- **Keyword Index**: `disable_torch_function.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/utils`):

- [`python_tuples.h_kw.md_docs.md`](./python_tuples.h_kw.md_docs.md)
- [`six.h_kw.md_docs.md`](./six.h_kw.md_docs.md)
- [`tensor_types.cpp_docs.md_docs.md`](./tensor_types.cpp_docs.md_docs.md)
- [`tensor_list.h_kw.md_docs.md`](./tensor_list.h_kw.md_docs.md)
- [`verbose.h_kw.md_docs.md`](./verbose.h_kw.md_docs.md)
- [`invalid_arguments.cpp_kw.md_docs.md`](./invalid_arguments.cpp_kw.md_docs.md)
- [`tensor_apply.h_kw.md_docs.md`](./tensor_apply.h_kw.md_docs.md)
- [`cuda_enabled.h_docs.md_docs.md`](./cuda_enabled.h_docs.md_docs.md)
- [`tensor_layouts.h_docs.md_docs.md`](./tensor_layouts.h_docs.md_docs.md)
- [`variadic.h_kw.md_docs.md`](./variadic.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `disable_torch_function.cpp_docs.md_docs.md`
- **Keyword Index**: `disable_torch_function.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
