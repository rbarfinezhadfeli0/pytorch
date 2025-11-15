# Documentation: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp_docs.md`
- **Size**: 5,900 bytes (5.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp`
- **Size**: 3,572 bytes (3.49 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

#include <runtime/OpenRegFunctions.h>

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR
static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_default_generator expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  return THPGenerator_initDefaultGenerator(
      at::globalContext().defaultGenerator(
          c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}
// LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR

// LITERALINCLUDE START: MODULE SET DEVICE HELPER

PyObject* _setDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackDeviceIndex(arg);
  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::openreg::set_device(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE END: MODULE SET DEVICE HELPER

PyObject* _exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::openreg::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDevice(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto device = static_cast<int32_t>(c10::openreg::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDeviceCount(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(c10::openreg::device_count());
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE START: OPENREG MODULE METHODS
static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_get_device", _getDevice, METH_NOARGS, nullptr},
    {"_set_device", _setDevice, METH_O, nullptr},
    {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
    {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};
// LITERALINCLUDE END: OPENREG MODULE METHODS
/*
 * When ASAN is enabled, PyTorch modifies the dlopen flag during import,
 * causing all global and weak symbols in _C.so and its dependent libraries
 * to be exposed to the global symbol scope, which in turn causes
 * subsequent symbols with the same name in other libraries to be intercepted.
 * Therefore, it cannot be named initModule here, otherwise initModule
 * in torch/csrc/Module.cpp will be called, resulting in failure.
 */
extern "C" OPENREG_EXPORT PyObject* initOpenRegModule(void) {
  static struct PyModuleDef openreg_C_module = {
      PyModuleDef_HEAD_INIT, "torch_openreg._C", nullptr, -1, methods};
  PyObject* mod = PyModule_Create(&openreg_C_module);

  return mod;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `PyModuleDef`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Context.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/utils.h`
- `torch/csrc/utils/device_lazy_init.h`
- `torch/csrc/utils/object_ptr.h`
- `torch/csrc/utils/python_numbers.h`
- `runtime/OpenRegFunctions.h`


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

This is a test file. Run it with:

```bash
python test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc`):

- [`stub.c_docs.md`](./stub.c_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)


## Cross-References

- **File Documentation**: `Module.cpp_docs.md`
- **Keyword Index**: `Module.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc`, which is part of the **core PyTorch library**.



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

This is a test file. Run it with:

```bash
python docs/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc`):

- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`stub.c_docs.md_docs.md`](./stub.c_docs.md_docs.md)
- [`stub.c_kw.md_docs.md`](./stub.c_kw.md_docs.md)
- [`Module.cpp_kw.md_docs.md`](./Module.cpp_kw.md_docs.md)
- [`CMakeLists.txt_kw.md_docs.md`](./CMakeLists.txt_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Module.cpp_docs.md_docs.md`
- **Keyword Index**: `Module.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
