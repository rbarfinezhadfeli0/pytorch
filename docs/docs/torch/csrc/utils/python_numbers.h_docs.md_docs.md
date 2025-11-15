# Documentation: `docs/torch/csrc/utils/python_numbers.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/python_numbers.h_docs.md`
- **Size**: 8,653 bytes (8.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/python_numbers.h`

## File Metadata

- **Path**: `torch/csrc/utils/python_numbers.h`
- **Size**: 6,129 bytes (5.99 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

// largest integer that can be represented consecutively in a double
const int64_t DOUBLE_INT_MAX = 9007199254740992;

inline PyObject* THPUtils_packDeviceIndex(c10::DeviceIndex value) {
  return PyLong_FromLong(value);
}

inline PyObject* THPUtils_packInt32(int32_t value) {
  return PyLong_FromLong(value);
}

inline PyObject* THPUtils_packInt64(int64_t value) {
  return PyLong_FromLongLong(value);
}

inline PyObject* THPUtils_packUInt32(uint32_t value) {
  return PyLong_FromUnsignedLong(value);
}

inline PyObject* THPUtils_packUInt64(uint64_t value) {
  return PyLong_FromUnsignedLongLong(value);
}

inline PyObject* THPUtils_packDoubleAsInt(double value) {
  return PyLong_FromDouble(value);
}

inline bool THPUtils_checkLongExact(PyObject* obj) {
  return PyLong_CheckExact(obj) && !PyBool_Check(obj);
}

inline bool THPUtils_checkLong(PyObject* obj) {
  // Fast path
  if (THPUtils_checkLongExact(obj)) {
    return true;
  }

#ifdef USE_NUMPY
  if (torch::utils::is_numpy_int(obj)) {
    return true;
  }
#endif

  return PyLong_Check(obj) && !PyBool_Check(obj);
}

inline int32_t THPUtils_unpackInt(PyObject* obj) {
  int overflow = 0;
  long value = PyLong_AsLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  TORCH_CHECK_VALUE(overflow == 0, "Overflow when unpacking long long");
  TORCH_CHECK_VALUE(
      value <= std::numeric_limits<int32_t>::max() &&
          value >= std::numeric_limits<int32_t>::min(),
      "Overflow when unpacking long");
  return (int32_t)value;
}

inline int64_t THPUtils_unpackLong(PyObject* obj) {
  int overflow = 0;
  long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  TORCH_CHECK_VALUE(overflow == 0, "Overflow when unpacking long long");
  return (int64_t)value;
}

inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {
  unsigned long value = PyLong_AsUnsignedLong(obj);
  if (PyErr_Occurred()) {
    throw python_error();
  }
  TORCH_CHECK_VALUE(
      value <= std::numeric_limits<uint32_t>::max(),
      "Overflow when unpacking long long");
  return (uint32_t)value;
}

inline uint64_t THPUtils_unpackUInt64(PyObject* obj) {
  unsigned long long value = PyLong_AsUnsignedLongLong(obj);
  if (PyErr_Occurred()) {
    throw python_error();
  }
  return (uint64_t)value;
}

bool THPUtils_checkIndex(PyObject* obj);

inline int64_t THPUtils_unpackIndex(PyObject* obj) {
  if (!THPUtils_checkLong(obj)) {
    auto index = THPObjectPtr(PyNumber_Index(obj));
    if (index == nullptr) {
      throw python_error();
    }
    // NB: This needs to be called before `index` goes out of scope and the
    // underlying object's refcount is decremented
    return THPUtils_unpackLong(index.get());
  }
  return THPUtils_unpackLong(obj);
}

inline bool THPUtils_unpackBool(PyObject* obj) {
  if (obj == Py_True) {
    return true;
  } else if (obj == Py_False) {
    return false;
  } else {
    TORCH_CHECK(false, "couldn't convert python object to boolean");
  }
}

inline bool THPUtils_checkBool(PyObject* obj) {
#ifdef USE_NUMPY
  if (torch::utils::is_numpy_bool(obj)) {
    return true;
  }
#endif
  return PyBool_Check(obj);
}

inline bool THPUtils_checkDouble(PyObject* obj) {
#ifdef USE_NUMPY
  if (torch::utils::is_numpy_scalar(obj)) {
    return true;
  }
#endif
  return PyFloat_Check(obj) || PyLong_Check(obj);
}

inline double THPUtils_unpackDouble(PyObject* obj) {
  if (PyFloat_Check(obj)) {
    return PyFloat_AS_DOUBLE(obj);
  }
  double value = PyFloat_AsDouble(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  return value;
}

inline c10::complex<double> THPUtils_unpackComplexDouble(PyObject* obj) {
  Py_complex value = PyComplex_AsCComplex(obj);
  if (value.real == -1.0 && PyErr_Occurred()) {
    throw python_error();
  }

  return c10::complex<double>(value.real, value.imag);
}

inline bool THPUtils_unpackNumberAsBool(PyObject* obj) {
#ifdef USE_NUMPY
  // Handle NumPy boolean scalars (np.bool_)
  if (torch::utils::is_numpy_bool(obj)) {
    int truth = PyObject_IsTrue(obj);
    if (truth == -1) {
      throw python_error();
    }
    return truth != 0;
  }
#endif
  if (PyFloat_Check(obj)) {
    return (bool)PyFloat_AS_DOUBLE(obj);
  }

  if (PyComplex_Check(obj)) {
    double real_val = PyComplex_RealAsDouble(obj);
    double imag_val = PyComplex_ImagAsDouble(obj);
    return !(real_val == 0 && imag_val == 0);
  }

  int overflow = 0;
  long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  // No need to check overflow, because when overflow occurred, it should
  // return true in order to keep the same behavior of numpy.
  return (bool)value;
}

inline c10::DeviceIndex THPUtils_unpackDeviceIndex(PyObject* obj) {
  int overflow = 0;
  long value = PyLong_AsLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  TORCH_CHECK(overflow == 0, "Overflow when unpacking DeviceIndex");
  TORCH_CHECK(
      value <= std::numeric_limits<c10::DeviceIndex>::max() &&
          value >= std::numeric_limits<c10::DeviceIndex>::min(),
      "Overflow when unpacking DeviceIndex");
  return (c10::DeviceIndex)value;
}

template <typename T>
inline T THPUtils_unpackInteger(PyObject* obj) {
  int overflow = -1;
  const auto value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  if (!overflow) {
    return static_cast<int64_t>(value);
  }
  // try unsigned
  const auto uvalue = PyLong_AsUnsignedLongLong(obj);
  if (uvalue == static_cast<std::decay_t<decltype(uvalue)>>(-1) &&
      PyErr_Occurred()) {
    throw python_error();
  }
  return static_cast<uint64_t>(uvalue);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 47 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Device.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/jit/frontend/tracer.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/object_ptr.h`
- `torch/csrc/utils/tensor_numpy.h`
- `cstdint`
- `limits`
- `stdexcept`


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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`disable_torch_function.cpp_docs.md`](./disable_torch_function.cpp_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `python_numbers.h_docs.md`
- **Keyword Index**: `python_numbers.h_kw.md`
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

- **File Documentation**: `python_numbers.h_docs.md_docs.md`
- **Keyword Index**: `python_numbers.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
