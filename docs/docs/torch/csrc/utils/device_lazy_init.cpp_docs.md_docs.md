# Documentation: `docs/torch/csrc/utils/device_lazy_init.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/device_lazy_init.cpp_docs.md`
- **Size**: 5,459 bytes (5.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/device_lazy_init.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/device_lazy_init.cpp`
- **Size**: 2,828 bytes (2.76 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/utils/device_lazy_init.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

#ifndef WIN32
#include <pthread.h>
#endif

namespace torch::utils {
namespace {

std::array<bool, at::COMPILE_TIME_MAX_DEVICE_TYPES> is_initialized{};
std::array<bool, at::COMPILE_TIME_MAX_DEVICE_TYPES> is_in_bad_fork{};
std::array<c10::once_flag, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    at_fork_once_flags{};

} // anonymous namespace

bool is_device_initialized(at::DeviceType device_type) {
  pybind11::gil_scoped_acquire g;
  return is_initialized[static_cast<int>(device_type)];
}

void device_lazy_init(at::DeviceType device_type) {
  pybind11::gil_scoped_acquire g;
  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (is_device_initialized(device_type)) {
    return;
  }

  auto maybe_mode = c10::impl::TorchDispatchModeTLS::get_mode(
      c10::impl::TorchDispatchModeKey::FAKE);
  if (maybe_mode) {
    return;
  }

  std::string module_name = "torch." + at::DeviceTypeName(device_type, true);
  auto module = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
  if (!module) {
    throw python_error();
  }

  if (device_type == at::DeviceType::PrivateUse1) {
    auto has_lazy_init_method =
        PyObject_HasAttrString(module.get(), "_lazy_init") == 1;
    if (!has_lazy_init_method) {
      return;
    }
  }

  auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
  if (!res) {
    throw python_error();
  }

  is_initialized[static_cast<int>(device_type)] = true;
}

void set_requires_device_init(at::DeviceType device_type, bool value) {
  is_initialized[static_cast<int>(device_type)] = !value;
}

bool is_device_in_bad_fork(at::DeviceType device_type) {
  return is_in_bad_fork[static_cast<int>(device_type)];
}

void set_device_in_bad_fork(at::DeviceType device_type, bool value) {
  is_in_bad_fork[static_cast<int>(device_type)] = value;
}

// Should be called before the first device runtime call.
void register_fork_handler_for_device_init(at::DeviceType device_type) {
#ifndef WIN32
  auto& flag = at_fork_once_flags[static_cast<int>(device_type)];
  c10::call_once(flag, [device_type]() {
    static at::DeviceType at_fork_device_type = device_type;
    pthread_atfork(nullptr, nullptr, []() {
      set_device_in_bad_fork(at_fork_device_type, true);
      if (is_device_lazy_init_supported(at_fork_device_type)) {
        set_requires_device_init(at_fork_device_type, true);
      }
    });
  });
#endif
}

} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `bool`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/impl/TorchDispatchModeTLS.h`
- `c10/util/CallOnce.h`
- `torch/csrc/utils/device_lazy_init.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/object_ptr.h`
- `pthread.h`


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

- **File Documentation**: `device_lazy_init.cpp_docs.md`
- **Keyword Index**: `device_lazy_init.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `device_lazy_init.cpp_docs.md_docs.md`
- **Keyword Index**: `device_lazy_init.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
