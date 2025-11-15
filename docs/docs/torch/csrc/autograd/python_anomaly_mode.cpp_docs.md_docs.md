# Documentation: `docs/torch/csrc/autograd/python_anomaly_mode.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/python_anomaly_mode.cpp_docs.md`
- **Size**: 6,845 bytes (6.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/python_anomaly_mode.cpp`

## File Metadata

- **Path**: `torch/csrc/autograd/python_anomaly_mode.cpp`
- **Size**: 4,181 bytes (4.08 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/Exception.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch::autograd {

void PyAnomalyMetadata::store_stack() {
  pybind11::gil_scoped_acquire gil;
  THPObjectPtr mod(PyImport_ImportModule("torch.fx.traceback"));
  if (!mod) {
    throw python_error();
  }

  THPObjectPtr list(PyObject_CallMethod(mod.get(), "format_stack", ""));
  if (!list) {
    throw python_error();
  }

  if (PyDict_SetItemString(dict(), ANOMALY_TRACE_KEY, list.get())) {
    throw python_error();
  }
}

void PyAnomalyMetadata::print_stack(const std::string& current_node_name) {
  pybind11::gil_scoped_acquire gil;
  if (!PyDict_Check(dict())) {
    TORCH_CHECK(false, "Anomaly metadata is not a python dictionary.");
  }
  PyObject* trace_stack = nullptr;
  if (PyDict_GetItemStringRef(dict(), ANOMALY_TRACE_KEY, &trace_stack) < 0) {
    throw python_error();
  }
  _print_stack(trace_stack, current_node_name, false);
  PyObject* pyparent = nullptr;
  if (PyDict_GetItemStringRef(dict(), ANOMALY_PARENT_KEY, &pyparent) < 0) {
    throw python_error();
  }

  // if there is no "parent_" in metadata, then it means this metadata's node
  // is the root and stop printing the traceback
  while (pyparent) {
    THPObjectPtr parent_metadata(PyObject_GetAttrString(pyparent, "metadata"));
    if (!parent_metadata) {
      throw python_error();
    }
    THPObjectPtr parent_name_pyobj(PyObject_CallMethod(pyparent, "name", ""));
    if (!parent_name_pyobj) {
      throw python_error();
    }
    const char* parent_name_char = PyUnicode_AsUTF8(parent_name_pyobj.get());
    if (!parent_name_char) {
      throw python_error();
    }
    const std::string parent_name(parent_name_char);
    PyObject* parent_stack = nullptr;
    if (PyDict_GetItemStringRef(
            parent_metadata.get(), ANOMALY_TRACE_KEY, &parent_stack) < 0) {
      throw python_error();
    }
    _print_stack(parent_stack, parent_name, true);
    // get the parent of this node, if this node is a root, pyparent is simply
    // null
    if (PyDict_GetItemStringRef(
            parent_metadata.get(), ANOMALY_PARENT_KEY, &pyparent) < 0) {
      throw python_error();
    }
  }
}

void PyAnomalyMetadata::assign_parent(
    const std::shared_ptr<Node>& parent_node) {
  // assign the python object of parent_node in metadata["parent_"]
  // if parent_node is nullptr, then do nothing (it can mean that "parent_" key
  // is not in metadata)

  pybind11::gil_scoped_acquire gil;
  if (!parent_node)
    return;

  THPObjectPtr parent_node_(functionToPyObject(parent_node));
  if (!parent_node_) {
    throw python_error();
  }
  if (PyDict_SetItemString(dict(), ANOMALY_PARENT_KEY, parent_node_.get())) {
    throw python_error();
  }
}

void _print_stack(
    PyObject* stack,
    const std::string& current_node_name,
    bool is_parent) {
  if (!stack) {
    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "No forward pass information available. Enable detect anomaly "
        "during forward pass for more information.");
    return;
  }

  THPObjectPtr empty_string(PyUnicode_FromString(""));
  if (!empty_string) {
    throw python_error();
  }

  // stack is a list of Python strings ending with newlines. Use join to convert
  // to a single string.
  THPObjectPtr msg(PyUnicode_Join(empty_string, stack));
  if (!msg) {
    throw python_error();
  }

  if (!is_parent) {
    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "Traceback of forward call that caused the error:\n",
        THPUtils_unpackString(msg.get()));
  } else {
    TORCH_WARN(
        "\n\n",
        "Previous calculation was induced by ",
        current_node_name,
        ". "
        "Traceback of forward call that induced the previous calculation:\n",
        THPUtils_unpackString(msg.get()));
  }
}

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `pybind11/pybind11.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/autograd/python_anomaly_mode.h`
- `torch/csrc/autograd/python_cpp_function.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/object_ptr.h`
- `torch/csrc/utils/pybind.h`
- `torch/csrc/utils/python_strings.h`


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

Files in the same folder (`torch/csrc/autograd`):

- [`graph_task.h_docs.md`](./graph_task.h_docs.md)
- [`python_function.cpp_docs.md`](./python_function.cpp_docs.md)
- [`profiler.h_docs.md`](./profiler.h_docs.md)
- [`TraceTypeManual.cpp_docs.md`](./TraceTypeManual.cpp_docs.md)
- [`python_autograd.h_docs.md`](./python_autograd.h_docs.md)
- [`variable_info.cpp_docs.md`](./variable_info.cpp_docs.md)
- [`jit_decomp_interface.h_docs.md`](./jit_decomp_interface.h_docs.md)
- [`input_buffer.cpp_docs.md`](./input_buffer.cpp_docs.md)
- [`python_variable.h_docs.md`](./python_variable.h_docs.md)
- [`python_nn_functions.h_docs.md`](./python_nn_functions.h_docs.md)


## Cross-References

- **File Documentation**: `python_anomaly_mode.cpp_docs.md`
- **Keyword Index**: `python_anomaly_mode.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/autograd`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/autograd`):

- [`python_cpp_function.h_kw.md_docs.md`](./python_cpp_function.h_kw.md_docs.md)
- [`anomaly_mode.cpp_kw.md_docs.md`](./anomaly_mode.cpp_kw.md_docs.md)
- [`python_nested_functions_manual.cpp_kw.md_docs.md`](./python_nested_functions_manual.cpp_kw.md_docs.md)
- [`variable_info.h_docs.md_docs.md`](./variable_info.h_docs.md_docs.md)
- [`python_nn_functions.h_docs.md_docs.md`](./python_nn_functions.h_docs.md_docs.md)
- [`python_cpp_function.h_docs.md_docs.md`](./python_cpp_function.h_docs.md_docs.md)
- [`profiler_legacy.cpp_kw.md_docs.md`](./profiler_legacy.cpp_kw.md_docs.md)
- [`saved_variable.cpp_docs.md_docs.md`](./saved_variable.cpp_docs.md_docs.md)
- [`python_fft_functions.h_docs.md_docs.md`](./python_fft_functions.h_docs.md_docs.md)
- [`python_autograd.h_kw.md_docs.md`](./python_autograd.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `python_anomaly_mode.cpp_docs.md_docs.md`
- **Keyword Index**: `python_anomaly_mode.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
