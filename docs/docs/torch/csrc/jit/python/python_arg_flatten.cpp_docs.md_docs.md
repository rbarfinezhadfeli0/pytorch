# Documentation: `docs/torch/csrc/jit/python/python_arg_flatten.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/python/python_arg_flatten.cpp_docs.md`
- **Size**: 9,605 bytes (9.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/python/python_arg_flatten.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/python/python_arg_flatten.cpp`
- **Size**: 6,967 bytes (6.80 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/irange.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/six.h>

#include <torch/csrc/autograd/grad_mode.h>

namespace torch::jit::python {

using namespace torch::autograd;
using namespace at;

// Alphabet used to describe structure of inputs/outputs (D for desc)
namespace D {
static constexpr char DictOpen = '<';
static constexpr char DictClose = '>';
static constexpr char ListOpen = '[';
static constexpr char ListClose = ']';
static constexpr char TupleOpen = '(';
static constexpr char TupleClose = ')';
static constexpr char Variable = 'v';
static constexpr char Bool = 'b';
static constexpr char Long = 'l';
static constexpr char Double = 'd';
static constexpr char String = 's';
static constexpr char NoneType = 'n';
} // namespace D

namespace {

inline bool PyNone_Check(PyObject* o) {
  return o == Py_None;
}

template <typename T>
py::object cast_handle_sequence(std::vector<py::handle> objs) {
  auto num_objs = objs.size();
  T sequence{num_objs};
  for (const auto i : c10::irange(num_objs)) {
    sequence[i] = py::reinterpret_borrow<py::object>(objs[i]);
  }
  return sequence;
}

void flatten_rec(PyObject* obj, ParsedArgs& args) {
  auto& structure = args.desc.structure;
  if (six::isTuple(obj)) {
    structure.push_back(D::TupleOpen);
    for (auto item : py::reinterpret_borrow<py::tuple>(obj))
      flatten_rec(item.ptr(), args);
    structure.push_back(D::TupleClose);
  } else if (PyList_Check(obj)) {
    structure.push_back(D::ListOpen);
    for (auto item : py::reinterpret_borrow<py::list>(obj))
      flatten_rec(item.ptr(), args);
    structure.push_back(D::ListClose);
  } else if (PyDict_Check(obj)) {
    auto* dict_items = PyDict_Items(obj);
    structure.push_back(D::DictOpen);
    for (auto item : py::reinterpret_borrow<py::list>(dict_items)) {
      flatten_rec(item.ptr(), args);
    }
    structure.push_back(D::DictClose);
    Py_DECREF(dict_items);
  } else if (THPUtils_checkString(obj)) {
    args.desc.strings.emplace_back(THPUtils_unpackString(obj));
    args.desc.structure.push_back(D::String);
  } else if (THPVariable_Check(obj)) {
    auto& var = THPVariable_Unpack(obj);
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Variable);
  } else if (PyNone_Check(obj)) {
    args.desc.structure.push_back(D::NoneType);
  } else if (PyBool_Check(obj)) { // Wrap bools in Bool tensors
    at::Tensor var = scalar_to_tensor(at::Scalar(THPUtils_unpackBool(obj)));
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Bool);
  } else if (PyLong_Check(obj)) { // Wrap longs in Long tensors
    at::Tensor var = scalar_to_tensor(at::Scalar(THPUtils_unpackLong(obj)));
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Long);
  } else if (PyFloat_Check(obj)) { // Wrap floats in Double tensors
    at::Tensor var = scalar_to_tensor(THPUtils_unpackDouble(obj));
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Double);
  } else {
    std::string msg =
        "Only tuples, lists and Variables are supported as JIT inputs/outputs. "
        "Dictionaries and strings are also accepted, but their usage is not "
        "recommended. Here, received an input of unsupported type: ";
    msg += THPUtils_typename(obj);
    throw std::runtime_error(msg);
  }
}

} // anonymous namespace

ParsedArgs flatten(py::handle obj) {
  ParsedArgs args;
  args.desc.grad_enabled = autograd::GradMode::is_enabled();
  flatten_rec(obj.ptr(), args);
  return args;
}

namespace {

template <typename T>
py::object cast_sequence(std::vector<py::object> objs) {
  auto num_objs = objs.size();
  T sequence{num_objs};
  for (const auto i : c10::irange(num_objs)) {
    sequence[i] = std::move(objs[i]);
  }
#if C10_RETURN_MOVE_IF_OLD_COMPILER
  return std::move(sequence);
#else
  return sequence;
#endif
}

py::object cast_dict(std::vector<py::object> objs) {
  auto num_objs = objs.size();
  py::dict sequence = {};
  for (const auto i : c10::irange(num_objs)) {
    py::tuple obj = py::reinterpret_borrow<py::tuple>(objs[i]);
    sequence[obj[0]] = obj[1];
  }
#if C10_RETURN_MOVE_IF_OLD_COMPILER
  return std::move(sequence);
#else
  return sequence;
#endif
}

py::object unflatten_rec(
    ArrayRef<Variable>::iterator& var_it,
    ArrayRef<Variable>::iterator& var_it_end,
    std::string::const_iterator& desc_it,
    std::vector<std::string>::const_iterator& str_it,
    std::vector<std::string>::const_iterator& str_it_end) {
  char type = *desc_it++;
  if (type == D::TupleOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::TupleClose)
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    ++desc_it;
    return cast_sequence<py::tuple>(objs);
  } else if (type == D::ListOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::ListClose)
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    ++desc_it;
    return cast_sequence<py::list>(objs);
  } else if (type == D::DictOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::DictClose) {
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    }
    ++desc_it;
    return cast_dict(objs);
  } else if (type == D::String) {
    if (str_it == str_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto str = *str_it++;
    return py::reinterpret_borrow<py::object>(THPUtils_packString(str));
  } else if (type == D::NoneType) {
    return py::reinterpret_borrow<py::object>(py::none());
  } else {
    // if (type == D::Long || type == D::Double || type == D::Bool ||
    // D::Variable) unwrap variables (D::Variable), or unwrap primitive types
    // (Long, Double, Bool) as variables for tracer.
    if (var_it == var_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto var = *var_it++;
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(var));
  }
}

} // anonymous namespace

PyObject* unflatten(ArrayRef<Variable> vars, const IODescriptor& desc) {
  // NB: We don't do correctness checking on descriptor.
  // It has to be a correct bytes object produced by unflatten.
  auto vars_it = vars.begin();
  auto vars_it_end = vars.end();
  auto desc_it = desc.structure.begin();
  std::vector<std::string>::const_iterator str_it = desc.strings.begin();
  std::vector<std::string>::const_iterator str_end = desc.strings.end();
  auto output = unflatten_rec(vars_it, vars_it_end, desc_it, str_it, str_end);
  if (vars_it != vars_it_end)
    throw std::runtime_error("Too many Variables given to unflatten");
  return output.release().ptr();
}

} // namespace torch::jit::python

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `PyObject`, `at`, `D`, `ParsedArgs`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `torch/csrc/jit/python/python_arg_flatten.h`
- `torch/csrc/utils/python_strings.h`
- `torch/csrc/utils/six.h`
- `torch/csrc/autograd/grad_mode.h`


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

Files in the same folder (`torch/csrc/jit/python`):

- [`python_ir.cpp_docs.md`](./python_ir.cpp_docs.md)
- [`python_custom_class.cpp_docs.md`](./python_custom_class.cpp_docs.md)
- [`python_ivalue.h_docs.md`](./python_ivalue.h_docs.md)
- [`pybind_utils.cpp_docs.md`](./pybind_utils.cpp_docs.md)
- [`opaque_obj.h_docs.md`](./opaque_obj.h_docs.md)
- [`update_graph_executor_opt.h_docs.md`](./update_graph_executor_opt.h_docs.md)
- [`python_tree_views.h_docs.md`](./python_tree_views.h_docs.md)
- [`python_ir.h_docs.md`](./python_ir.h_docs.md)
- [`python_tracer.h_docs.md`](./python_tracer.h_docs.md)
- [`python_arg_flatten.h_docs.md`](./python_arg_flatten.h_docs.md)


## Cross-References

- **File Documentation**: `python_arg_flatten.cpp_docs.md`
- **Keyword Index**: `python_arg_flatten.cpp_kw.md`
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

- **File Documentation**: `python_arg_flatten.cpp_docs.md_docs.md`
- **Keyword Index**: `python_arg_flatten.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
