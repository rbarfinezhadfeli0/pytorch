# Documentation: `docs/torch/csrc/jit/python/python_ivalue.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/python/python_ivalue.h_docs.md`
- **Size**: 6,288 bytes (6.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/python/python_ivalue.h`

## File Metadata

- **Path**: `torch/csrc/jit/python/python_ivalue.h`
- **Size**: 3,682 bytes (3.60 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/ivalue.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace c10::ivalue {

// concrete ivalue Holder that hold a py::object
struct C10_EXPORT ConcretePyObjectHolder final : PyObjectHolder {
 public:
  static c10::intrusive_ptr<PyObjectHolder> create(py::object py_obj) {
    return c10::make_intrusive<ConcretePyObjectHolder>(std::move(py_obj));
  }

  static c10::intrusive_ptr<PyObjectHolder> create(const py::handle& handle) {
    py::gil_scoped_acquire ag;
    return c10::make_intrusive<ConcretePyObjectHolder>(
        handle.cast<py::object>());
  }

  PyObject* getPyObject() override {
    return py_obj_.ptr();
  }

  InferredType tryToInferType() override {
    pybind11::gil_scoped_acquire ag;
    return torch::jit::tryToInferType(py_obj_);
  }

  IValue toIValue(const TypePtr& type, std::optional<int32_t> N = std::nullopt)
      override {
    pybind11::gil_scoped_acquire ag;
    return torch::jit::toIValue(py_obj_, type, N);
  }

  std::string toStr() override {
    pybind11::gil_scoped_acquire ag;
    return py::str(py_obj_);
  }

  std::vector<at::Tensor> extractTensors() override {
    // We could implement this entirely in C++ via pybind11 but it turns out to
    // be substantially slower. Namely, the total time taken by markCompleted on
    // a CUDAFuture is 21.5us with this implementation, but goes up to 58.7us
    // when using C++. The reason is unclear.
    try {
      pybind11::gil_scoped_acquire ag;

#if IS_PYBIND_2_13_PLUS
      PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
          storage;
      auto& extractorFn =
          storage
              .call_once_and_store_result([]() -> py::object {
                return py::module_::import("torch._jit_internal")
                    .attr("_extract_tensors");
              })
              .get_stored();
#else
      static py::object& extractorFn = *new py::object(
          py::module::import("torch._jit_internal").attr("_extract_tensors"));
#endif

      return extractorFn(py_obj_).cast<std::vector<at::Tensor>>();
    } catch (py::error_already_set& e) {
      auto err = std::runtime_error(
          c10::str("Cannot extract tensors from value: ", e.what()));
      {
        pybind11::gil_scoped_acquire ag;
        e.restore();
        PyErr_Clear();
      }
      throw std::runtime_error(err);
    }
  }

  // Note [Destructing py::object]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // (1) Why py_obj_ = py::none(); does not work. Because we also need to
  // acquire GIL when destructing py::object of None that de-references None.
  // https://docs.python.org/3/c-api/none.html#c.Py_RETURN_NONE
  //
  // https://stackoverflow.com/questions/15287590/why-should-py-increfpy-none-be-required-before-returning-py-none-in-c
  //
  // (2) Why we need to call dec_ref() explicitly. Because py::object of
  // nullptr, on destruction, effectively does nothing because of it calls
  // Py_XDECREF(NULL) underlying.
  // https://docs.python.org/3/c-api/refcounting.html#c.Py_XDECREF
  ~ConcretePyObjectHolder() override {
    pybind11::gil_scoped_acquire ag;
    py_obj_.dec_ref();
    // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
    // decref on the PyObject again.
    py_obj_.ptr() = nullptr;
  }

  // explicit construction to avoid erroneous implicit conversion and
  // copy-initialization
  explicit ConcretePyObjectHolder(py::object py_obj)
      : py_obj_(std::move(py_obj)) {}

 private:
  py::object py_obj_;
};

} // namespace c10::ivalue

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `py`, `c10`

**Classes/Structs**: `C10_EXPORT`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `pybind11/pybind11.h`
- `torch/csrc/jit/python/pybind_utils.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/pybind.h`


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

Files in the same folder (`torch/csrc/jit/python`):

- [`python_ir.cpp_docs.md`](./python_ir.cpp_docs.md)
- [`python_custom_class.cpp_docs.md`](./python_custom_class.cpp_docs.md)
- [`pybind_utils.cpp_docs.md`](./pybind_utils.cpp_docs.md)
- [`opaque_obj.h_docs.md`](./opaque_obj.h_docs.md)
- [`update_graph_executor_opt.h_docs.md`](./update_graph_executor_opt.h_docs.md)
- [`python_tree_views.h_docs.md`](./python_tree_views.h_docs.md)
- [`python_ir.h_docs.md`](./python_ir.h_docs.md)
- [`python_tracer.h_docs.md`](./python_tracer.h_docs.md)
- [`python_arg_flatten.h_docs.md`](./python_arg_flatten.h_docs.md)


## Cross-References

- **File Documentation**: `python_ivalue.h_docs.md`
- **Keyword Index**: `python_ivalue.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/python`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/python`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/python`):

- [`opaque_obj.h_kw.md_docs.md`](./opaque_obj.h_kw.md_docs.md)
- [`script_init.h_docs.md_docs.md`](./script_init.h_docs.md_docs.md)
- [`python_tree_views.cpp_docs.md_docs.md`](./python_tree_views.cpp_docs.md_docs.md)
- [`python_dict.cpp_docs.md_docs.md`](./python_dict.cpp_docs.md_docs.md)
- [`python_tree_views.h_docs.md_docs.md`](./python_tree_views.h_docs.md_docs.md)
- [`opaque_obj.h_docs.md_docs.md`](./opaque_obj.h_docs.md_docs.md)
- [`python_custom_class.cpp_docs.md_docs.md`](./python_custom_class.cpp_docs.md_docs.md)
- [`python_tracer.cpp_kw.md_docs.md`](./python_tracer.cpp_kw.md_docs.md)
- [`python_interpreter.cpp_kw.md_docs.md`](./python_interpreter.cpp_kw.md_docs.md)
- [`python_tracer.cpp_docs.md_docs.md`](./python_tracer.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `python_ivalue.h_docs.md_docs.md`
- **Keyword Index**: `python_ivalue.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
