# Documentation: `docs/torch/csrc/utils/pybind.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/pybind.cpp_docs.md`
- **Size**: 7,596 bytes (7.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/pybind.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/pybind.cpp`
- **Size**: 5,213 bytes (5.09 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_symnode.h>

namespace pybind11::detail {

bool type_caster<c10::SymInt>::load(py::handle src, bool /*unused*/) {
  if (torch::is_symint(src)) {
    auto node = src.attr("node");
    if (py::isinstance<c10::SymNodeImpl>(node)) {
      value = c10::SymInt(py::cast<c10::SymNode>(node));
      return true;
    }

    value = c10::SymInt(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(node)));
    return true;
  }

  auto raw_obj = src.ptr();

  if (THPVariable_Check(raw_obj)) {
    auto& var = THPVariable_Unpack(raw_obj);
    if (var.numel() == 1 &&
        at::isIntegralType(var.dtype().toScalarType(), /*include_bool*/ true)) {
      auto scalar = var.item();
      TORCH_INTERNAL_ASSERT(scalar.isIntegral(/*include bool*/ false));
      value = scalar.toSymInt();
      return true;
    }
  }

  if (THPUtils_checkIndex(raw_obj)) {
    value = c10::SymInt{THPUtils_unpackIndex(raw_obj)};
    return true;
  }
  return false;
}

py::handle type_caster<c10::SymInt>::cast(
    const c10::SymInt& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (si.is_symbolic()) {
    auto* py_node = dynamic_cast<torch::impl::PythonSymNodeImpl*>(
        si.toSymNodeImplUnowned());
    if (py_node) {
      // Return the Python directly (unwrap)
      return torch::get_symint_class()(py_node->getPyObj()).release();
    } else {
      // Wrap the C++ into Python
      auto inner = py::cast(si.toSymNode());
      if (!inner) {
        throw python_error();
      }
      return torch::get_symint_class()(inner).release();
    }
  } else {
    auto m = si.maybe_as_int();
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return py::cast(m.value()).release();
  }
}

bool type_caster<c10::SymFloat>::load(py::handle src, bool /*unused*/) {
  if (torch::is_symfloat(src)) {
    value = c10::SymFloat(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(src.attr("node"))));
    return true;
  }

  auto raw_obj = src.ptr();
  if (THPUtils_checkDouble(raw_obj)) {
    value = c10::SymFloat{THPUtils_unpackDouble(raw_obj)};
    return true;
  }
  return false;
}

py::handle type_caster<c10::SymFloat>::cast(
    const c10::SymFloat& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (si.is_symbolic()) {
    // TODO: generalize this to work with C++ backed class
    auto* py_node =
        dynamic_cast<torch::impl::PythonSymNodeImpl*>(si.toSymNodeImpl().get());
    TORCH_INTERNAL_ASSERT(py_node);
    return torch::get_symfloat_class()(py_node->getPyObj()).release();
  } else {
    return py::cast(si.as_float_unchecked()).release();
  }
}

bool type_caster<c10::SymBool>::load(py::handle src, bool /*unused*/) {
  if (torch::is_symbool(src)) {
    value = c10::SymBool(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(src.attr("node"))));
    return true;
  }

  auto raw_obj = src.ptr();
  if (THPUtils_checkBool(raw_obj)) {
    value = c10::SymBool{THPUtils_unpackBool(raw_obj)};
    return true;
  }
  return false;
}

py::handle type_caster<c10::SymBool>::cast(
    const c10::SymBool& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (auto m = si.maybe_as_bool()) {
    return py::cast(*m).release();
  } else {
    // TODO: generalize this to work with C++ backed class
    auto* py_node =
        dynamic_cast<torch::impl::PythonSymNodeImpl*>(si.toSymNodeImpl().get());
    TORCH_INTERNAL_ASSERT(py_node);
    return torch::get_symbool_class()(py_node->getPyObj()).release();
  }
}

bool type_caster<c10::Scalar>::load(py::handle src, bool /*unused*/) {
  TORCH_INTERNAL_ASSERT(
      0, "pybind11 loading for c10::Scalar NYI (file a bug if you need it)");
}

py::handle type_caster<c10::Scalar>::cast(
    const c10::Scalar& scalar,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (scalar.isIntegral(/*includeBool*/ false)) {
    // We have to be careful here; we cannot unconditionally route through
    // SymInt because integer data from Tensors can easily be MIN_INT or
    // very negative, which conflicts with the allocated range.
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymInt()).release();
    } else {
      if (scalar.type() == at::ScalarType::UInt64) {
        return py::cast(scalar.toUInt64()).release();
      } else {
        return py::cast(scalar.toLong()).release();
      }
    }
  } else if (scalar.isFloatingPoint()) {
    // This isn't strictly necessary but we add it for symmetry
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymFloat()).release();
    } else {
      return py::cast(scalar.toDouble()).release();
    }
  } else if (scalar.isBoolean()) {
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymBool()).release();
    }
    return py::cast(scalar.toBool()).release();
  } else if (scalar.isComplex()) {
    return py::cast(scalar.toComplexDouble()).release();
  } else {
    TORCH_INTERNAL_ASSERT(0, "unrecognized scalar type ", scalar.type());
  }
}

} // namespace pybind11::detail

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `pybind11`

**Classes/Structs**: `auto`, `auto`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/utils/pybind.h`
- `torch/csrc/utils/python_arg_parser.h`
- `torch/csrc/utils/python_symnode.h`


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

- **File Documentation**: `pybind.cpp_docs.md`
- **Keyword Index**: `pybind.cpp_kw.md`
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

- **File Documentation**: `pybind.cpp_docs.md_docs.md`
- **Keyword Index**: `pybind.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
