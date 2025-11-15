# Documentation: `docs/torch/csrc/autograd/functions/utils.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/functions/utils.h_docs.md`
- **Size**: 5,685 bytes (5.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/functions/utils.h`

## File Metadata

- **Path**: `torch/csrc/autograd/functions/utils.h`
- **Size**: 3,230 bytes (3.15 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/InferenceMode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/core/Tensor.h>

#include <functional>
#include <memory>
#include <vector>

namespace torch::autograd {

using function_constructor = std::function<std::shared_ptr<Node>(edge_list&&)>;

/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
TORCH_API variable_list wrap_outputs(
    const variable_list& inputs,
    tensor_list&& outputs,
    const function_constructor& ctr);

///  Checks that inputs contains exactly `args` items and that the first
///  `required_args`
/// items are not nullptr. If not specified, `required_args` defaults to `args`.
TORCH_API void check_input_variables(
    const char* name,
    const variable_list& inputs,
    int args,
    int required_args = -1,
    bool allow_undefined = false);

struct ComputeRequiresGrad : IterArgs<ComputeRequiresGrad> {
  bool out = false;
  using IterArgs<ComputeRequiresGrad>::operator();
  void operator()(const at::Tensor& tensor) {
    const auto& var = static_cast<const Variable&>(tensor);
    if (var.defined() && var.requires_grad()) {
      out = true;
    }
  }
  void operator()(const std::optional<at::Tensor>& tensor) {
    if (tensor.has_value()) {
      (*this)(*tensor);
    }
  }
  bool short_circuit() {
    return out;
  }
};

template <typename... Args>
inline bool compute_requires_grad(Args&&... args) {
  if (!GradMode::is_enabled()) {
    return false;
  }
  return ComputeRequiresGrad().apply(std::forward<Args>(args)...).out;
}

inline void set_history(
    const at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  TORCH_CHECK(grad_fn != nullptr);
  if (variable.defined()) {
    // If the codegen triggers this, you most likely want to add your newly
    // added function to the DONT_REQUIRE_DERIVATIVE list in
    // tools/autograd/gen_variable_type.py
    TORCH_CHECK(
        isDifferentiableType(variable.scalar_type()),
        "Autograd not support dtype: ",
        variable.scalar_type());
    auto output_nr = grad_fn->add_input_metadata(variable);
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}

inline void set_history(
    const std::vector<Variable>& variables,
    const std::shared_ptr<Node>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}

inline bool isFwGradDefined(const std::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() && t->_fw_grad(/*level */ 0).defined();
}

inline bool isFwGradDefinedTensorList(const at::ITensorListRef& variables) {
  bool ret = false;
  for (auto& variable : variables) {
    ret |= isFwGradDefined(variable);
  }
  return ret;
}

inline bool isFwGradDefinedTensorList(
    const c10::List<std::optional<at::Tensor>>& li) {
  bool ret = false;
  for (auto i : c10::irange(li.size())) {
    auto t = li.get(i);
    ret |= isFwGradDefined(t);
  }
  return ret;
}

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ComputeRequiresGrad`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd/functions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/csrc/autograd/InferenceMode.h`
- `torch/csrc/autograd/autograd.h`
- `torch/csrc/autograd/function.h`
- `torch/csrc/autograd/variable.h`
- `torch/csrc/utils/variadic.h`
- `ATen/core/Tensor.h`
- `functional`
- `memory`
- `vector`


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

Files in the same folder (`torch/csrc/autograd/functions`):

- [`basic_ops.cpp_docs.md`](./basic_ops.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`tensor.cpp_docs.md`](./tensor.cpp_docs.md)
- [`tensor.h_docs.md`](./tensor.h_docs.md)
- [`accumulate_grad.h_docs.md`](./accumulate_grad.h_docs.md)
- [`init.cpp_docs.md`](./init.cpp_docs.md)
- [`comm.h_docs.md`](./comm.h_docs.md)
- [`comm.cpp_docs.md`](./comm.cpp_docs.md)
- [`basic_ops.h_docs.md`](./basic_ops.h_docs.md)


## Cross-References

- **File Documentation**: `utils.h_docs.md`
- **Keyword Index**: `utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/autograd/functions`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/autograd/functions`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/autograd/functions`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`basic_ops.cpp_kw.md_docs.md`](./basic_ops.cpp_kw.md_docs.md)
- [`accumulate_grad.h_kw.md_docs.md`](./accumulate_grad.h_kw.md_docs.md)
- [`pybind.h_docs.md_docs.md`](./pybind.h_docs.md_docs.md)
- [`utils.h_kw.md_docs.md`](./utils.h_kw.md_docs.md)
- [`accumulate_grad.h_docs.md_docs.md`](./accumulate_grad.h_docs.md_docs.md)
- [`init.cpp_kw.md_docs.md`](./init.cpp_kw.md_docs.md)
- [`basic_ops.h_kw.md_docs.md`](./basic_ops.h_kw.md_docs.md)
- [`init.cpp_docs.md_docs.md`](./init.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `utils.h_docs.md_docs.md`
- **Keyword Index**: `utils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
