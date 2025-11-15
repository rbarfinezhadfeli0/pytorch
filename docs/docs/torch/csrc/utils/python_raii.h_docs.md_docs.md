# Documentation: `docs/torch/csrc/utils/python_raii.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/python_raii.h_docs.md`
- **Size**: 5,032 bytes (4.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/python_raii.h`

## File Metadata

- **Path**: `torch/csrc/utils/python_raii.h`
- **Size**: 2,658 bytes (2.60 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#include <torch/csrc/utils/pybind.h>
#include <optional>
#include <tuple>

namespace torch::impl {

template <typename GuardT, typename... Args>
struct RAIIContextManager {
  explicit RAIIContextManager(Args&&... args)
      : args_(std::forward<Args>(args)...) {}

  void enter() {
    auto emplace = [&](Args... args) {
      guard_.emplace(std::forward<Args>(args)...);
    };
    std::apply(std::move(emplace), args_);
  }

  void exit() {
    guard_ = std::nullopt;
  }

 private:
  std::optional<GuardT> guard_;
  std::tuple<Args...> args_;
};

// Turns a C++ RAII guard into a Python context manager.
// See _ExcludeDispatchKeyGuard in python_dispatch.cpp for example.
template <typename GuardT, typename... GuardArgs>
void py_context_manager(const py::module& m, const char* name) {
  using ContextManagerT = RAIIContextManager<GuardT, GuardArgs...>;
  py::class_<ContextManagerT>(m, name)
      .def(py::init<GuardArgs...>())
      .def("__enter__", [](ContextManagerT& guard) { guard.enter(); })
      .def(
          "__exit__",
          [](ContextManagerT& guard,
             const py::object& exc_type,
             const py::object& exc_value,
             const py::object& traceback) { guard.exit(); });
}

template <typename GuardT, typename... Args>
struct DeprecatedRAIIContextManager {
  explicit DeprecatedRAIIContextManager(Args&&... args) {
    guard_.emplace(std::forward<Args>(args)...);
  }

  void enter() {}

  void exit() {
    guard_ = std::nullopt;
  }

 private:
  std::optional<GuardT> guard_;
  std::tuple<Args...> args_;
};

// Definition: a "Python RAII guard" is an object in Python that acquires
// a resource on init and releases the resource on deletion.
//
// This API turns a C++ RAII guard into an object can be used either as a
// Python context manager or as a "Python RAII guard".
//
// Please prefer `py_context_manager` to this API if you are binding a new
// RAII guard into Python because "Python RAII guards" don't work as expected
// in Python (Python makes no guarantees about when an object gets deleted)
template <typename GuardT, typename... GuardArgs>
void py_context_manager_DEPRECATED(const py::module& m, const char* name) {
  using ContextManagerT = DeprecatedRAIIContextManager<GuardT, GuardArgs...>;
  py::class_<ContextManagerT>(m, name)
      .def(py::init<GuardArgs...>())
      .def("__enter__", [](ContextManagerT& guard) { guard.enter(); })
      .def(
          "__exit__",
          [](ContextManagerT& guard,
             const py::object& exc_type,
             const py::object& exc_value,
             const py::object& traceback) { guard.exit(); });
}

} // namespace torch::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `RAIIContextManager`, `DeprecatedRAIIContextManager`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/utils/pybind.h`
- `optional`
- `tuple`


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

- **File Documentation**: `python_raii.h_docs.md`
- **Keyword Index**: `python_raii.h_kw.md`
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

- **File Documentation**: `python_raii.h_docs.md_docs.md`
- **Keyword Index**: `python_raii.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
