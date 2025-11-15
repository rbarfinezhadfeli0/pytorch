# Documentation: `docs/torch/csrc/Dtype.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/Dtype.cpp_docs.md`
- **Size**: 8,812 bytes (8.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/Dtype.cpp`

## File Metadata

- **Path**: `torch/csrc/Dtype.cpp`
- **Size**: 6,340 bytes (6.19 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/Dtype.h>

#include <c10/core/ScalarType.h>
#include <structmember.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/tensor_types.h>
#include <cstring>

PyObject* THPDtype_New(at::ScalarType scalar_type, const std::string& name) {
  HANDLE_TH_ERRORS
  AT_ASSERT(name.length() < DTYPE_NAME_LEN);
  auto type = &THPDtypeType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPDtype*>(self.get());
  self_->scalar_type = scalar_type;
  std::strncpy(self_->name, name.c_str(), DTYPE_NAME_LEN);
  return self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_is_floating_point(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isFloatingType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_itemsize(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(
      scalarTypeToTypeMeta(self->scalar_type).itemsize());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_is_complex(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isComplexType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_is_signed(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isSignedType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_reduce(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  /*
   * For singletons, a string is returned. The string should be interpreted
   * as the name of a global variable.
   */
  auto self = reinterpret_cast<THPDtype*>(_self);
  return THPUtils_packString(self->name);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_to_real(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = reinterpret_cast<THPDtype*>(_self);
  auto scalar_type = self->scalar_type;
  if (!at::isFloatingType(self->scalar_type)) {
    scalar_type = at::toRealValueType(self->scalar_type);
  }
  return Py_NewRef(torch::getTHPDtype(scalar_type));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_to_complex(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = reinterpret_cast<THPDtype*>(_self);
  auto scalar_type = self->scalar_type;
  if (!at::isComplexType(self->scalar_type)) {
    scalar_type = at::toComplexType(self->scalar_type);
  }
  return Py_NewRef(torch::getTHPDtype(scalar_type));
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);

static const std::initializer_list<PyGetSetDef> THPDtype_properties = {
    {"is_floating_point",
     reinterpret_cast<getter>(THPDtype_is_floating_point),
     nullptr,
     nullptr,
     nullptr},
    {"is_complex",
     reinterpret_cast<getter>(THPDtype_is_complex),
     nullptr,
     nullptr,
     nullptr},
    {"is_signed",
     reinterpret_cast<getter>(THPDtype_is_signed),
     nullptr,
     nullptr,
     nullptr},
    {"itemsize",
     reinterpret_cast<getter>(THPDtype_itemsize),
     nullptr,
     nullptr,
     nullptr},
    {nullptr}};

static const std::initializer_list<PyMethodDef> THPDtype_methods = {
    {"__reduce__", THPDtype_reduce, METH_NOARGS, nullptr},
    {"to_real", THPDtype_to_real, METH_NOARGS, nullptr},
    {"to_complex", THPDtype_to_complex, METH_NOARGS, nullptr},
    {nullptr} /* Sentinel */
};

static PyObject* THPDtype_repr(THPDtype* self) {
  return THPUtils_packString(std::string("torch.") + self->name);
}

PyTypeObject THPDtypeType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch.dtype", /* tp_name */
    sizeof(THPDtype), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    reinterpret_cast<reprfunc>(THPDtype_repr), /* tp_repr */
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
    // NOLINTNEXTLINE(*const-cast)
    const_cast<PyMethodDef*>(std::data(THPDtype_methods)), /* tp_methods */
    nullptr, /* tp_members */
    // NOLINTNEXTLINE(*const-cast)
    const_cast<PyGetSetDef*>(std::data(THPDtype_properties)), /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

void THPDtype_init(PyObject* module) {
  // Set a __dict__ with `__module__` = `torch`. This means
  // `__module__` value will be inherited by instances
  // (i.e. `torch.float32.__module__ == "torch"`). This will prevent
  // Pickle from having to search all of sys.modules in order to find
  // the module when pickling a dtype instance.
  //
  // We have to do this in C++ because extension types are not mutable
  // from Python code.
  //
  // See https://github.com/pytorch/pytorch/issues/65077
  TORCH_INTERNAL_ASSERT(THPDtypeType.tp_dict == nullptr);
  auto dict = THPObjectPtr(PyDict_New());
  if (!dict)
    throw python_error();
  auto torch = THPUtils_packString("torch");
  if (!torch)
    throw python_error();
  if (PyDict_SetItemString(dict, "__module__", torch) < 0) {
    throw python_error();
  }
  THPDtypeType.tp_dict = dict.release();

  if (PyType_Ready(&THPDtypeType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPDtypeType);
  if (PyModule_AddObject(
          module, "dtype", reinterpret_cast<PyObject*>(&THPDtypeType)) != 0) {
    throw python_error();
  }
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Dtype.h`
- `c10/core/ScalarType.h`
- `structmember.h`
- `torch/csrc/DynamicTypes.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/utils/object_ptr.h`
- `torch/csrc/utils/python_numbers.h`
- `torch/csrc/utils/python_strings.h`
- `torch/csrc/utils/pythoncapi_compat.h`
- `torch/csrc/utils/tensor_dtypes.h`
- `torch/csrc/utils/tensor_types.h`
- `cstring`


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

Files in the same folder (`torch/csrc`):

- [`itt_wrapper.cpp_docs.md`](./itt_wrapper.cpp_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`Export.h_docs.md`](./Export.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`Size.h_docs.md`](./Size.h_docs.md)
- [`stub.c_docs.md`](./stub.c_docs.md)
- [`Device.h_docs.md`](./Device.h_docs.md)
- [`Layout.h_docs.md`](./Layout.h_docs.md)
- [`Exceptions.h_docs.md`](./Exceptions.h_docs.md)
- [`PyInterpreter.h_docs.md`](./PyInterpreter.h_docs.md)


## Cross-References

- **File Documentation**: `Dtype.cpp_docs.md`
- **Keyword Index**: `Dtype.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc`):

- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`Exceptions.cpp_docs.md_docs.md`](./Exceptions.cpp_docs.md_docs.md)
- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`Exceptions.h_docs.md_docs.md`](./Exceptions.h_docs.md_docs.md)
- [`serialization.cpp_kw.md_docs.md`](./serialization.cpp_kw.md_docs.md)
- [`QScheme.cpp_kw.md_docs.md`](./QScheme.cpp_kw.md_docs.md)
- [`DataLoader.cpp_kw.md_docs.md`](./DataLoader.cpp_kw.md_docs.md)
- [`Size.h_docs.md_docs.md`](./Size.h_docs.md_docs.md)
- [`DeviceAccelerator.h_kw.md_docs.md`](./DeviceAccelerator.h_kw.md_docs.md)
- [`Device.cpp_kw.md_docs.md`](./Device.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Dtype.cpp_docs.md_docs.md`
- **Keyword Index**: `Dtype.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
