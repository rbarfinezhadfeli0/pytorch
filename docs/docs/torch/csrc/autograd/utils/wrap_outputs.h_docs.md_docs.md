# Documentation: `docs/torch/csrc/autograd/utils/wrap_outputs.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/utils/wrap_outputs.h_docs.md`
- **Size**: 6,235 bytes (6.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/utils/wrap_outputs.h`

## File Metadata

- **Path**: `torch/csrc/autograd/utils/wrap_outputs.h`
- **Size**: 3,743 bytes (3.66 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// Wrap tensor operation outputs as PyObject*

#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <torch/csrc/python_headers.h>
#include <initializer_list>
#include <tuple>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_qschemes.h>

namespace torch::autograd::utils {

inline PyObject* wrap(bool value) {
  if (value) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

inline PyObject* wrap(c10::DeviceIndex value) {
  return THPUtils_packDeviceIndex(value);
}

inline PyObject* wrap(int64_t value) {
  return THPUtils_packInt64(value);
}

inline PyObject* wrap(double value) {
  return PyFloat_FromDouble(value);
}

inline PyObject* wrap(c10::complex<double> value) {
  // I could probably also use FromComplex with a reinterpret cast,
  // but... eh.
  return PyComplex_FromDoubles(value.real(), value.imag());
}

inline PyObject* wrap(void* value) {
  return PyLong_FromVoidPtr(value);
}

inline PyObject* wrap(THPDtype* dtype) {
  return Py_NewRef(dtype);
}

inline PyObject* wrap(at::ScalarType scalarType) {
  return Py_NewRef(getTHPDtype(scalarType));
}

inline PyObject* wrap(THPLayout* layout) {
  return Py_NewRef(layout);
}

inline PyObject* wrap(at::Layout layout) {
  return Py_NewRef(getTHPLayout(layout));
}

inline PyObject* wrap(const at::Tensor& tensor) {
  return THPVariable_Wrap(tensor);
}

inline PyObject* wrap(const at::Scalar& scalar) {
  return wrap(scalar_to_tensor(scalar));
}

inline PyObject* wrap(at::QScheme qscheme) {
  auto* thp_qscheme = torch::utils::getTHPQScheme(qscheme);
  Py_INCREF(thp_qscheme);
  return thp_qscheme;
}

inline PyObject* wrap(at::TensorList tl) {
  auto r = THPObjectPtr{PyTuple_New(static_cast<Py_ssize_t>(tl.size()))};
  if (!r)
    throw python_error();
  for (const auto i : c10::irange(tl.size())) {
    PyTuple_SET_ITEM(r.get(), i, wrap(tl[i]));
  }
  return r.release();
}

inline PyObject* wrap(at::IntArrayRef list) {
  auto r = THPObjectPtr{PyTuple_New(static_cast<Py_ssize_t>(list.size()))};
  if (!r)
    throw python_error();
  for (const auto i : c10::irange(list.size())) {
    PyTuple_SET_ITEM(r.get(), i, wrap(list[i]));
  }
  return r.release();
}

inline PyObject* wrap(at::Stream stream) {
  return THPStream_Wrap(stream);
}

namespace detail {
template <typename F, typename Tuple, size_t... Is>
void apply_with_idx_impl(
    const F& f,
    Tuple& t,
    std::index_sequence<Is...> /*indices*/) {
  (void)std::initializer_list<int>{(f(std::get<Is>(t), Is), 0)...};
}

// For tuple(a, b, c), calls f(a, 0), f(b, 1), f(c, 2)
template <typename F, typename... Ts>
void apply_with_idx(const F& f, std::tuple<Ts...>& t) {
  apply_with_idx_impl(f, t, std::index_sequence_for<Ts...>{});
}
} // namespace detail

template <typename... Ts>
PyObject* wrap(std::tuple<Ts...> values) {
  auto r = THPObjectPtr{PyTuple_New(sizeof...(Ts))};
  if (!r)
    throw python_error();
  detail::apply_with_idx(
      [&](auto& value, size_t idx) {
        PyTuple_SET_ITEM(r.get(), idx, wrap(std::move(value)));
      },
      values);
  return r.release();
}

template <typename... Ts>
PyObject* wrap(PyTypeObject* type, std::tuple<Ts...> values) {
  auto r = THPObjectPtr{PyStructSequence_New(type)};
  if (!r)
    throw python_error();
  detail::apply_with_idx(
      [&](auto& value, size_t idx) {
        PyStructSequence_SET_ITEM(r.get(), idx, wrap(std::move(value)));
      },
      values);
  return r.release();
}

} // namespace torch::autograd::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ScalarOps.h`
- `ATen/core/Tensor.h`
- `c10/util/irange.h`
- `torch/csrc/python_headers.h`
- `initializer_list`
- `tuple`
- `torch/csrc/Dtype.h`
- `torch/csrc/DynamicTypes.h`
- `torch/csrc/Layout.h`
- `torch/csrc/QScheme.h`
- `torch/csrc/autograd/python_variable.h`
- `torch/csrc/autograd/variable.h`
- `torch/csrc/utils/python_numbers.h`
- `torch/csrc/utils/tensor_qschemes.h`


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

Files in the same folder (`torch/csrc/autograd/utils`):

- [`warnings.cpp_docs.md`](./warnings.cpp_docs.md)
- [`error_messages.h_docs.md`](./error_messages.h_docs.md)
- [`warnings.h_docs.md`](./warnings.h_docs.md)
- [`grad_layout_contract.h_docs.md`](./grad_layout_contract.h_docs.md)
- [`lambda_post_hook.h_docs.md`](./lambda_post_hook.h_docs.md)
- [`python_arg_parsing.h_docs.md`](./python_arg_parsing.h_docs.md)


## Cross-References

- **File Documentation**: `wrap_outputs.h_docs.md`
- **Keyword Index**: `wrap_outputs.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/autograd/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/autograd/utils`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/autograd/utils`):

- [`grad_layout_contract.h_docs.md_docs.md`](./grad_layout_contract.h_docs.md_docs.md)
- [`error_messages.h_kw.md_docs.md`](./error_messages.h_kw.md_docs.md)
- [`error_messages.h_docs.md_docs.md`](./error_messages.h_docs.md_docs.md)
- [`lambda_post_hook.h_kw.md_docs.md`](./lambda_post_hook.h_kw.md_docs.md)
- [`warnings.cpp_kw.md_docs.md`](./warnings.cpp_kw.md_docs.md)
- [`python_arg_parsing.h_docs.md_docs.md`](./python_arg_parsing.h_docs.md_docs.md)
- [`grad_layout_contract.h_kw.md_docs.md`](./grad_layout_contract.h_kw.md_docs.md)
- [`python_arg_parsing.h_kw.md_docs.md`](./python_arg_parsing.h_kw.md_docs.md)
- [`warnings.cpp_docs.md_docs.md`](./warnings.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `wrap_outputs.h_docs.md_docs.md`
- **Keyword Index**: `wrap_outputs.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
