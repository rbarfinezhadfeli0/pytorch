# Documentation: `docs/torch/csrc/utils/tensor_list.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/tensor_list.cpp_docs.md`
- **Size**: 4,861 bytes (4.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/tensor_list.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/tensor_list.cpp`
- **Size**: 2,347 bytes (2.29 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/functorch/TensorWrapper.h>
#include <torch/csrc/utils/tensor_list.h>

#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_scalars.h>

using namespace at;

namespace torch::utils {

static PyObject* recursive_to_list(
    const char* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t dim,
    ScalarType scalarType,
    size_t elementSize) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  if (dim == ndim) {
    return torch::utils::load_scalar(data, scalarType);
  }
  auto n = sizes[dim];
  auto list = THPObjectPtr(PyList_New(n));
  if (!list)
    throw python_error();
  for (const auto i : c10::irange(n)) {
    PyObject* obj = recursive_to_list(
        data, sizes, strides, dim + 1, scalarType, elementSize);
    if (!obj)
      throw python_error();
    PyList_SET_ITEM(list.get(), i, obj);
    auto advance_data_ptr = strides[dim] * elementSize;
    TORCH_INTERNAL_ASSERT(data || (advance_data_ptr == 0));
    data += advance_data_ptr;
  }
  return list.release();
}

const Tensor& recursive_unwrap(const Tensor& tensor) {
  if (auto* wrapper = at::functorch::maybeGetTensorWrapper(tensor))
    return recursive_unwrap(wrapper->value());
  return tensor;
}

PyObject* tensor_to_list(const Tensor& tensor) {
  {
    py::object pytensor =
        py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));
    TORCH_CHECK(
        !tensor.unsafeGetTensorImpl()->is_python_dispatch(),
        ".tolist() is not supported for tensor subclasses, got ",
        Py_TYPE(pytensor.ptr())->tp_name);
  }
  // check if it is a grad tracking tensor and unwrap.
  Tensor data = tensor.resolve_conj().resolve_neg();
  data = recursive_unwrap(data);
  if (!data.device().is_cpu()) {
    pybind11::gil_scoped_release no_gil;
    data = data.toBackend(Backend::CPU);
  }
  TORCH_CHECK(
      tensor.numel() == 0 || data.const_data_ptr(),
      "tolist() shouldn't be called on a tensor with unallocated storage");
  return recursive_to_list(
      (const char*)data.const_data_ptr(),
      data.sizes(),
      data.strides(),
      0,
      data.scalar_type(),
      tensor.numel() == 0 ? 0 : data.dtype().itemsize());
}

} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/functorch/TensorWrapper.h`
- `torch/csrc/utils/tensor_list.h`
- `c10/util/irange.h`
- `pybind11/pybind11.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/autograd/python_variable.h`
- `torch/csrc/utils/pybind.h`
- `torch/csrc/utils/python_scalars.h`


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

- **File Documentation**: `tensor_list.cpp_docs.md`
- **Keyword Index**: `tensor_list.cpp_kw.md`
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

- **File Documentation**: `tensor_list.cpp_docs.md_docs.md`
- **Keyword Index**: `tensor_list.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
