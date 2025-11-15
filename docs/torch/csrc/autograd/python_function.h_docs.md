# Documentation: `torch/csrc/autograd/python_function.h`

## File Metadata

- **Path**: `torch/csrc/autograd/python_function.h`
- **Size**: 5,371 bytes (5.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/object_ptr.h>

#include <c10/core/DeviceGuard.h>
#include <optional>

#include <memory>
#include <vector>

namespace torch::jit {
struct Graph;
}

namespace torch::autograd {

// A Function which is implemented by a Python object (i.e., a THPFunction).
// Calls to 'apply' are forwarded to the Python method implementation.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct PyNode : public Node {
  PyNode(THPObjectPtr obj) : obj(obj.release()) {}

  PyObject* to_py_args(
      const variable_list& inputs,
      at::OptionalDeviceGuard* device_guard);
  variable_list to_variable_list(
      const PyObject* r,
      const std::vector<bool>& is_variable_input);

  variable_list apply(variable_list&& inputs) override;
  variable_list apply_with_saved_impl(
      const variable_list& inputs,
      const SwapSavedVariables& saved);

  void release_variables() override;
  std::string name() const override;
  bool is_traceable() override;

  bool is_aot_backward() const override;

  void compiled_args(CompiledNodeArgs& args) const override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  // THPFunction this Function is wrapping.  Owning!
  PyObject* obj;

  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~PyNode() override {
    // Can't use THPObjectPtr as a field in this class; destructor won't take
    // out GIL!  When I forgot to do this by hand
    // TestAutograd.test_inplace_view_python called me out about it.
    // If python is already dead, leak the wrapped python objects
    if (Py_IsInitialized()) {
      pybind11::gil_scoped_acquire gil;
      Py_DECREF(obj);
    }
  }
};

/**
 * Cast an object into a tuple, if it is not a tuple already. Returns true
 * if the original object was not a tuple.
 */
inline bool ensure_tuple(THPObjectPtr& obj) {
  if (PyTuple_Check(obj.get()))
    return false;

  PyObject* tuple = PyTuple_New(1);
  if (!tuple)
    throw python_error();
  PyTuple_SET_ITEM(tuple, 0, obj.release());
  obj = tuple;
  return true;
}

} // namespace torch::autograd

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPFunction {
  PyObject_HEAD

  PyObject* needs_input_grad;

  // Python tuple of tensors whose variables we should save.  Set
  // by Python with 'save_for_backward'.  If nullptr, no tensors were
  // saved.
  PyObject* to_save;
  // Python tuple of tensors which are not differentiable.  Set by
  // Python with 'mark_non_differentiable'.  If nullptr, no tensors were
  // non-differentiable.
  PyObject* non_differentiable;
  // Python tuple of tensors which had inplace updates in the forward()
  // pass.  Set by Python with 'mark_dirty'.  If nullptr, no tensors were
  // modified inplace.
  PyObject* dirty_tensors;

  // boolean indicating whether to materialize undefined output grad tensors
  // into tensors full of zeros. Set by Python with 'set_materialize_grads'.
  // Default is true.
  bool materialize_grads;

  // boolean indicating whether the function is a "pure view", meaning that
  // replaying the view is enough to get a correct backward.
  bool pure_view;

  // boolean indicating whether to materialize output grad tensors
  // corresponding to non-differentiable outputs. Normally, someone would
  // already get this behavior by switching off materialize_grads,
  // but there are certain use cases where that is not feasible:
  // https://github.com/pytorch/pytorch/pull/98659#pullrequestreview-1376822560
  bool materialize_non_diff_grads;

  PyObject* compiled_autograd_backward_state;
  std::vector<c10::SymInt> compiled_autograd_symints;

  std::vector<torch::autograd::VariableInfo> output_info;
  std::vector<torch::autograd::VariableInfo> input_info;
  std::vector<torch::autograd::SavedVariable> saved_variables;
  // For each input, true if the input is a THPVariable
  std::vector<bool> is_variable_input;
  char has_freed_buffers;

  PyObject* saved_for_forward;
  // The actual PyNode (in the autograd graph) that this data was
  // saved for.  This field may be NULL (because a user can construct
  // a THPFunction directly from Python), but when this field is non-NULL,
  // it is guaranteed that cdata.lock()->obj == this
  //
  // In most ordinary use, this field should always be non-NULL; e.g.,
  // when we allocate a THPFunction because we are running Node.apply,
  // after constructing a THPFunction, we immediately allocate a PyNode
  // for it.  We can't enforce this directly in the constructor of
  // THPFunction though, because there's no way to keep it live long enough
  // to save an owning reference to PyNode into the grad_fn of a Variable.
  std::weak_ptr<torch::autograd::PyNode> cdata;
};

bool THPFunction_initModule(PyObject* module);
TORCH_PYTHON_API extern PyTypeObject THPFunctionType;
TORCH_PYTHON_API extern PyObject* THPFunctionClass;
TORCH_PYTHON_API extern PyObject* THPGradientEdgeClass;

inline bool THPFunction_Check(PyObject* obj) {
  return PyObject_IsInstance(obj, (PyObject*)&THPFunctionType);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Graph`, `PyNode`, `THPFunction`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/python_headers.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/Export.h`
- `torch/csrc/autograd/custom_function.h`
- `torch/csrc/autograd/function.h`
- `torch/csrc/autograd/saved_variable.h`
- `torch/csrc/autograd/variable.h`
- `torch/csrc/utils/object_ptr.h`
- `c10/core/DeviceGuard.h`
- `optional`
- `memory`
- `vector`


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

- **File Documentation**: `python_function.h_docs.md`
- **Keyword Index**: `python_function.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
