# Documentation: `docs/torch/csrc/autograd/python_cpp_function.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/python_cpp_function.cpp_docs.md`
- **Size**: 15,009 bytes (14.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/python_cpp_function.cpp`

## File Metadata

- **Path**: `torch/csrc/autograd/python_cpp_function.cpp`
- **Size**: 12,050 bytes (11.77 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/irange.h>
#include <torch/csrc/autograd/python_cpp_function.h>

#include <torch/csrc/python_headers.h>
#include <cstdio>
#include <memory>
#include <typeindex>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

using namespace torch::autograd;

namespace torch::autograd {

namespace {

PyObject* THPCppFunction_call(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  if (kwargs && PyDict_Size(kwargs) != 0) {
    return PyErr_Format(PyExc_TypeError, "keyword arguments are not supported");
  }

  auto num_inputs = PyTuple_GET_SIZE(args);
  auto num_inputs_required = ((THPCppFunction*)self)->cdata->num_inputs();
  if (num_inputs != num_inputs_required) {
    return PyErr_Format(
        PyExc_TypeError,
        "expected %d arguments, got %d instead",
        num_inputs_required,
        num_inputs);
  }
  variable_list vars(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    PyObject* arg = PyTuple_GET_ITEM(args, i);
    if (arg == Py_None) {
      continue;
    }
    if (!THPVariable_Check(arg)) {
      return PyErr_Format(PyExc_TypeError, "argument %d is not a Variable", i);
    }
    vars[i] = THPVariable_Unpack(arg);
  }

  variable_list output;

  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release nogil;
    output = (*((THPCppFunction*)self)->cdata)(std::move(vars));
  }
  END_HANDLE_TH_ERRORS

  auto num_outputs = output.size();
  if (num_outputs == 1) {
    // assume we want to unpack one element tuples for now
    return THPVariable_Wrap(output[0]);
  }

  THPObjectPtr tuple(PyTuple_New(static_cast<Py_ssize_t>(num_outputs)));
  for (size_t i = 0; i != num_outputs; ++i) {
    PyTuple_SET_ITEM(tuple.get(), i, THPVariable_Wrap(output[i]));
  }
  return tuple.release();
}

int THPCppFunction_traverse(PyObject* self, visitproc visit, void* arg) {
  if ((((THPCppFunction*)self)->cdata).use_count() == 1) {
    // The fields traversed below are owned by the cpp grad_fn, which we own a
    // reference to. We should only them traverse however if we are the only
    // owner of the grad_fn, otherwise we risk prematurely gc'ing the grad_fn.
    //
    // See: https://github.com/pytorch/pytorch/issues/102174
    auto& fn = *((THPCppFunction*)self)->cdata;
    for (const auto& hook : fn.tensor_pre_hooks()) {
      if (auto pyhook = dynamic_cast<PyFunctionTensorPreHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
    // NOTE [retains_grad_hook PyObject traversal]
    // In theory this shouldn't be necessary, because retains_grad_hooks should
    // not contain any PyFunctionTensorPreHooks. The alternative is to have a
    // check that actually guarantees this.
    for (const auto& pair : fn.retains_grad_hooks()) {
      if (auto pyhook =
              dynamic_cast<PyFunctionTensorPreHook*>(pair.second.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
    for (const auto& hook : fn.pre_hooks()) {
      if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
    for (const auto& hook : fn.post_hooks()) {
      if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
  }
  return 0;
}

int THPCppFunction_clear(PyObject* self) {
  auto f = (THPCppFunction*)self;
  // Remove the weak ref of the c++ object if it exist
  if (f->cdata) {
    f->cdata->set_pyobj(nullptr);
  }
  f->cdata.reset();
  return 0;
}

void THPCppFunction_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  THPCppFunction_clear(self);
  ((THPCppFunction*)self)->cdata.~shared_ptr();
  Py_TYPE(self)->tp_free(self);
}

} // namespace

PyObject* THPCppFunction_next_functions(PyObject* self, void* _unused) {
  auto cdata = reinterpret_cast<const THPCppFunction*>(self)->cdata;
  const auto num_next = cdata->num_outputs();
  THPObjectPtr py_functions(PyTuple_New(num_next));
  if (!py_functions)
    return nullptr;
  for (const auto i : c10::irange(num_next)) {
    auto& c_tuple = cdata->next_edge(i);
    THPObjectPtr tuple(PyTuple_New(2));
    if (!tuple)
      return nullptr;
    PyObject* py_fn = functionToPyObject(c_tuple.function);
    if (!py_fn)
      return nullptr;
    PyTuple_SET_ITEM(tuple.get(), 0, py_fn);
    PyObject* py_idx = THPUtils_packUInt32(c_tuple.input_nr);
    if (!py_idx)
      return nullptr;
    PyTuple_SET_ITEM(tuple.get(), 1, py_idx);
    PyTuple_SET_ITEM(py_functions.get(), i, tuple.release());
  }
  return py_functions.release();
}

PyObject* THPCppFunction_metadata(PyObject* self, void* _unused) {
  auto* metadata =
      static_cast<PyAnomalyMetadata*>(
          reinterpret_cast<THPCppFunction*>(self)->cdata->metadata())
          ->dict();

  Py_XINCREF(metadata);
  return metadata;
}

PyObject* THPCppFunction_requires_grad(PyObject* self, void* unused) {
  Py_RETURN_TRUE;
}

PyObject* THPCppFunction_register_hook_dict(PyObject* self, PyObject* _var) {
  if (!THPVariable_Check(_var)) {
    return PyErr_Format(
        PyExc_TypeError, "_register_hook_dict expected a variable");
  }
  auto var = (THPVariable*)_var;
  auto& fn = *((THPCppFunction*)self)->cdata;
  fn.add_tensor_pre_hook(std::make_unique<PyFunctionTensorPreHook>(
      var->backward_hooks, THPVariable_Unpack(var).output_nr()));
  Py_RETURN_NONE;
}

PyObject* THPCppFunction_register_hook(PyObject* self, PyObject* hook) {
  auto& fn = *((THPCppFunction*)self)->cdata;
  return registerFunctionHook(fn, hook);
}

PyObject* THPCppFunction_register_prehook(PyObject* self, PyObject* hook) {
  auto& fn = *((THPCppFunction*)self)->cdata;
  return registerFunctionPreHook(fn, hook);
}

PyObject* THPCppFunction_name(PyObject* self, PyObject* noargs) {
  auto& fn = *((THPCppFunction*)self)->cdata;
  return THPUtils_packString(fn.name());
}

PyObject* THPCppFunction_sequence_nr(PyObject* self, PyObject* noargs) {
  auto& fn = *((THPCppFunction*)self)->cdata;
  return THPUtils_packUInt64(fn.sequence_nr());
}

static PyObject* THPCppFunction_set_sequence_nr(
    PyObject* self,
    PyObject* sequence_nr) {
  HANDLE_TH_ERRORS
  auto& fn = *((THPCppFunction*)self)->cdata;
  fn.set_sequence_nr(THPUtils_unpackUInt64(sequence_nr));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCppFunction_input_metadata(PyObject* self, void* closure) {
  HANDLE_TH_ERRORS;
  auto& fn = *((THPCppFunction*)self)->cdata;
  const auto num_inputs =
      fn.num_inputs(); // Assuming there's a method to get the number of inputs
  THPObjectPtr list(PyTuple_New(num_inputs));
  if (!list) {
    return nullptr;
  }
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& metadata = fn.input_metadata(i);
    THPObjectPtr item(py::cast(metadata).release().ptr());
    if (!item) {
      return nullptr;
    }
    PyTuple_SET_ITEM(list.get(), i, item.release());
  }
  return list.release();
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyMethodDef default_methods[] = {
    THP_FUNCTION_DEFAULT_METHODS,
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyGetSetDef default_properties[] = {
    THP_FUNCTION_DEFAULT_PROPERTIES,
    {nullptr}};

PyTypeObject* _initFunctionPyTypeObject(
    PyTypeObject& type,
    const char* name,
    PyGetSetDef* function_properties,
    PyMethodDef* function_methods) {
  type.ob_base = {
    PyObject_HEAD_INIT(nullptr)
      0};
  // NOLINTNEXTLINE(misc-redundant-expression)
  type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
  type.tp_name = name;
  type.tp_basicsize = sizeof(THPCppFunction);
  type.tp_call = THPCppFunction_call;
  type.tp_methods = function_methods ? function_methods : default_methods;
  type.tp_getset =
      function_properties ? function_properties : default_properties;
  type.tp_dealloc = THPCppFunction_dealloc;
  type.tp_traverse = THPCppFunction_traverse;
  type.tp_clear = THPCppFunction_clear;
  if (PyType_Ready(&type) < 0) {
    TORCH_CHECK(false, "Unable to instantiate PyTypeObject for ", name);
  }
  return &type;
}

static std::unordered_map<std::type_index, THPObjectPtr> cpp_function_types_map;
static std::unordered_set<PyTypeObject*> cpp_function_types_set;

struct DefaultFunctionType {
  DefaultFunctionType() : type() {
    _initFunctionPyTypeObject(type, "CppFunction", nullptr, nullptr);
  }

  PyTypeObject type;
};

static PyTypeObject* get_default_type() {
  static DefaultFunctionType default_type;
  return &(default_type.type);
}

PyObject* functionToPyObject(const std::shared_ptr<Node>& cdata) {
  if (!cdata) {
    Py_RETURN_NONE;
  }

  if (auto pfw = dynamic_cast<PyNode*>(cdata.get())) {
    PyObject* obj = pfw->obj;
    Py_INCREF(obj);
    return obj;
  }

  if (cdata->pyobj()) {
    Py_INCREF(cdata->pyobj());
  } else {
    auto& fn = *cdata;
    auto it = cpp_function_types_map.find(std::type_index(typeid(fn)));
    PyTypeObject* type = nullptr;
    if (it == cpp_function_types_map.end()) {
      type = get_default_type();
    } else {
      type = (PyTypeObject*)it->second.get();
    }

    THPObjectPtr obj(type->tp_alloc(type, 0));
    if (!obj)
      return nullptr;
    THPCppFunction* f = (THPCppFunction*)obj.get();
    new (&f->cdata) std::shared_ptr<Node>(cdata);

    // No INCREF here as we only have a weak reference
    cdata->set_pyobj(obj.release());
  }

  return cdata->pyobj();
}

void registerCppFunction(const std::type_info& type, PyTypeObject* pytype) {
  Py_INCREF((PyObject*)pytype);
  cpp_function_types_map[std::type_index(type)] =
      THPObjectPtr((PyObject*)pytype);
  cpp_function_types_set.insert(pytype);
}

bool THPCppFunction_Check(PyObject* obj) {
  THPObjectPtr type = THPObjectPtr(PyObject_Type(obj));
  if ((PyTypeObject*)type.get() == get_default_type()) {
    return true;
  }
  if (cpp_function_types_set.find((PyTypeObject*)type.get()) ==
      cpp_function_types_set.end()) {
    return false;
  } else {
    return true;
  }
}

static PyObject* callRegisterFn(PyObject* dict, PyObject* hook) {
  THPObjectPtr register_fn(
      PyObject_GetAttrString(THPFunctionClass, "_register_hook"));
  if (!register_fn) {
    return nullptr;
  }
  THPObjectPtr res(
      PyObject_CallFunctionObjArgs(register_fn.get(), dict, hook, nullptr));
  if (!res) {
    return nullptr;
  }
  return res.release();
}

PyObject* registerFunctionHook(Node& fn, PyObject* hook) {
  PyObject* dict = Py_None;
  for (const auto& hook : fn.post_hooks()) {
    if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
      dict = pyhook->dict;
      break;
    }
  }
  THPObjectPtr res{callRegisterFn(dict, hook)};
  if (!res) {
    return nullptr;
  }
  if (dict == Py_None) {
    dict = PyTuple_GET_ITEM(res.get(), 0);
    fn.add_post_hook(std::make_unique<PyFunctionPostHook>(dict));
  }

  PyObject* handle = PyTuple_GET_ITEM(res.get(), 1);
  Py_INCREF(handle);
  return handle;
}

// This is almost a copy of the function above except post -> pre
PyObject* registerFunctionPreHook(Node& fn, PyObject* hook) {
  PyObject* dict = Py_None;
  for (const auto& hook : fn.pre_hooks()) {
    if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
      dict = pyhook->dict;
      break;
    }
  }
  THPObjectPtr res{callRegisterFn(dict, hook)};
  if (!res) {
    return nullptr;
  }
  if (dict == Py_None) {
    dict = PyTuple_GET_ITEM(res.get(), 0);
    fn.add_pre_hook(std::make_unique<PyFunctionPreHook>(dict));
  }

  PyObject* handle = PyTuple_GET_ITEM(res.get(), 1);
  Py_INCREF(handle);
  return handle;
}

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 24 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `PyObject`

**Classes/Structs**: `PyMethodDef`, `PyGetSetDef`, `DefaultFunctionType`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `torch/csrc/autograd/python_cpp_function.h`
- `torch/csrc/python_headers.h`
- `cstdio`
- `memory`
- `typeindex`
- `unordered_map`
- `pybind11/pybind11.h`
- `torch/csrc/DynamicTypes.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/autograd/python_anomaly_mode.h`
- `torch/csrc/autograd/python_function.h`
- `torch/csrc/autograd/python_hook.h`
- `torch/csrc/autograd/python_variable.h`
- `torch/csrc/utils/pybind.h`
- `torch/csrc/utils/python_numbers.h`
- `torch/csrc/utils/python_strings.h`


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

- **File Documentation**: `python_cpp_function.cpp_docs.md`
- **Keyword Index**: `python_cpp_function.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/csrc/autograd`):

- [`python_cpp_function.h_kw.md_docs.md`](./python_cpp_function.h_kw.md_docs.md)
- [`anomaly_mode.cpp_kw.md_docs.md`](./anomaly_mode.cpp_kw.md_docs.md)
- [`python_nested_functions_manual.cpp_kw.md_docs.md`](./python_nested_functions_manual.cpp_kw.md_docs.md)
- [`variable_info.h_docs.md_docs.md`](./variable_info.h_docs.md_docs.md)
- [`python_nn_functions.h_docs.md_docs.md`](./python_nn_functions.h_docs.md_docs.md)
- [`python_cpp_function.h_docs.md_docs.md`](./python_cpp_function.h_docs.md_docs.md)
- [`profiler_legacy.cpp_kw.md_docs.md`](./profiler_legacy.cpp_kw.md_docs.md)
- [`saved_variable.cpp_docs.md_docs.md`](./saved_variable.cpp_docs.md_docs.md)
- [`python_fft_functions.h_docs.md_docs.md`](./python_fft_functions.h_docs.md_docs.md)
- [`python_autograd.h_kw.md_docs.md`](./python_autograd.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `python_cpp_function.cpp_docs.md_docs.md`
- **Keyword Index**: `python_cpp_function.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
