# Documentation: `docs/torch/csrc/autograd/functions/basic_ops.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/functions/basic_ops.h_docs.md`
- **Size**: 5,879 bytes (5.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/functions/basic_ops.h`

## File Metadata

- **Path**: `torch/csrc/autograd/functions/basic_ops.h`
- **Size**: 3,396 bytes (3.32 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/irange.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

#include <memory>
#include <string>
#include <vector>

namespace torch::autograd {

struct TORCH_API Error : public Node {
  Error(std::string msg, edge_list&& next_edges)
      : Node(std::move(next_edges)), msg(std::move(msg)) {}

  Error(std::string msg) : msg(std::move(msg)) {}

  variable_list apply(variable_list&& inputs) override;
  variable_list apply(variable_list&& inputs) const;

  void compiled_args(CompiledNodeArgs& args) const override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  std::string msg;
};

// We print grad_fn names in tensor printing. For functions with backward
// NYI, grad_fn=<Error> will be printed if we use Error, which is confusing. So
// special case with a new NotImplemented function here.
struct TORCH_API NotImplemented : public Error {
  NotImplemented(const std::string& forward_fn, edge_list&& next_edges)
      : Error(
            "derivative for " + forward_fn + " is not implemented",
            std::move(next_edges)) {}

  NotImplemented(const std::string& forward_fn)
      : Error("derivative for " + forward_fn + " is not implemented") {}
};

// Identity in forward, Error in backward. Used to implement
// @once_differentiable
struct TORCH_API DelayedError : public Node {
  DelayedError(std::string msg, int64_t num_inputs) : msg(std::move(msg)) {
    for ([[maybe_unused]] const auto _ [[maybe_unused]] :
         c10::irange(num_inputs)) {
      add_input_metadata(Node::undefined_input());
    }
  }

  variable_list apply(variable_list&& inputs) override;
  variable_list apply(variable_list&& inputs) const;

  std::string msg;
};

struct TORCH_API UndefinedGrad : public Node {
  UndefinedGrad() {
    add_input_metadata(Node::undefined_input());
  }

  variable_list apply(variable_list&& inputs) override;
  variable_list apply(variable_list&& inputs) const;
};

struct TORCH_API UndefinedGradBackward : public Node {
  UndefinedGradBackward(edge_list&& next_edges) : Node(std::move(next_edges)) {}

  UndefinedGradBackward() = default;

  variable_list apply(variable_list&& inputs) override;
  variable_list apply(variable_list&& inputs) const;

  void compiled_args(CompiledNodeArgs& args) const override {}
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override {
    return apply(variable_list(inputs));
  }
};

struct TORCH_API GraphRoot : public Node {
  GraphRoot(edge_list functions, variable_list inputs)
      : Node(std::move(functions)), outputs(std::move(inputs)) {
    // Ensures calls to stream() on a GraphRoot instance reflect current
    // stream(s) on devices of root grad tensors at the time the instance is
    // constructed.
    for (const auto& t : outputs) {
      add_input_metadata(t);
    }
  }

  variable_list apply(variable_list&& inputs) override {
    return outputs;
  }

  void compiled_args(CompiledNodeArgs& args) const override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  variable_list outputs;
};

struct TORCH_API Identity : public Node {
  variable_list apply(variable_list&& inputs) override;
};

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd/functions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `torch/csrc/Export.h`
- `torch/csrc/autograd/function.h`
- `torch/csrc/autograd/variable.h`
- `memory`
- `string`
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

Files in the same folder (`torch/csrc/autograd/functions`):

- [`basic_ops.cpp_docs.md`](./basic_ops.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`tensor.cpp_docs.md`](./tensor.cpp_docs.md)
- [`tensor.h_docs.md`](./tensor.h_docs.md)
- [`accumulate_grad.h_docs.md`](./accumulate_grad.h_docs.md)
- [`init.cpp_docs.md`](./init.cpp_docs.md)
- [`comm.h_docs.md`](./comm.h_docs.md)
- [`comm.cpp_docs.md`](./comm.cpp_docs.md)


## Cross-References

- **File Documentation**: `basic_ops.h_docs.md`
- **Keyword Index**: `basic_ops.h_kw.md`
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

Files in the same folder (`docs/torch/csrc/autograd/functions`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`basic_ops.cpp_kw.md_docs.md`](./basic_ops.cpp_kw.md_docs.md)
- [`accumulate_grad.h_kw.md_docs.md`](./accumulate_grad.h_kw.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`pybind.h_docs.md_docs.md`](./pybind.h_docs.md_docs.md)
- [`utils.h_kw.md_docs.md`](./utils.h_kw.md_docs.md)
- [`accumulate_grad.h_docs.md_docs.md`](./accumulate_grad.h_docs.md_docs.md)
- [`init.cpp_kw.md_docs.md`](./init.cpp_kw.md_docs.md)
- [`basic_ops.h_kw.md_docs.md`](./basic_ops.h_kw.md_docs.md)
- [`init.cpp_docs.md_docs.md`](./init.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `basic_ops.h_docs.md_docs.md`
- **Keyword Index**: `basic_ops.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
