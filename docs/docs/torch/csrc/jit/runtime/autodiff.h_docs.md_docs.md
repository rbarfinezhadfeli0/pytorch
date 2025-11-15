# Documentation: `docs/torch/csrc/jit/runtime/autodiff.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/autodiff.h_docs.md`
- **Size**: 6,486 bytes (6.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/autodiff.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/autodiff.h`
- **Size**: 3,930 bytes (3.84 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <vector>

namespace torch::jit {

using value_list = std::vector<Value*>;
// clang-format off
// Example showcasing how Gradient is constructed:
//
// Let's assume we have a function f, `m` and `n` do not require grad
// (`n` can depend only on `m`):
//   y, n = f(x, m)
//
// Now, let's assume that the reverse of f (called f') needs to use values of `x`, `t` and `y`.
// `t` is an intermediate value produced in the body of f, and let's assume that it requires
// grad too.
//
// In this case differentiate(f) will return this:
//   y, n, t = f(x, m)        // `t` is appended to the output list
//   dx = f'(dy, dt, x, t, y) // No `dm` or `dn` because they do not require gradient
//                            // All needed values from f are prepended to the input list
//
//   f_real_outputs = 2       // Only first two outputs were present in f originally
//   df_input_vjps = {0, 2}   // i.e. connect grad_fn of y and t variables produced by f,
//                    y  t    // with y's output_nr = 0 and t's output_nr = 1
//   df_input_captures = {I0, O2, O0} // Order matches the prefix of inputs to df
//                        x   t   y
//   df_output_vjps = {0}     // i.e. connect next_edge[0] of grad_fn to x's (grad_fn, output_nr).
//
// Terminology: vjp = vector-jacobian product
// clang-format on

struct Gradient {
  explicit operator bool() const {
    return df != nullptr;
  }
  std::shared_ptr<Graph> f;
  std::shared_ptr<Graph> df;

  // Describes how to construct outputs of f from what its graph will return.
  // This is necessary because some trailing outputs are intermediates produced
  // only to be saved for df (and should be ignored).
  size_t f_real_outputs = 0; // initialized for safety.

  // df inputs are split into two sections: vjps (aka grad_outputs) and
  // captures. VJPs are "seeds" for the gradient computation given for each
  // input capture of an Output kind. Captures are values the need to be saved
  // when f is run. We handle inputs specially, because this allows us to avoid
  // adding extra vjps as df inputs.

  std::vector<size_t> df_input_vjps; // Offsets into f's outputs.
  // capture can come from inputs or outputs
  std::vector<size_t> df_input_captured_inputs; // Offsets into f's inputs
  std::vector<size_t> df_input_captured_outputs; // Offsets into f's outputs

  // df will produce vjps for a subset of inputs of f that required grad.
  // df_output_vjps[idx] == inp_idx means that idx-th output of df produces a
  // vjp for inp_idx-th input of f.
  std::vector<size_t> df_output_vjps; // Offsets into f's inputs.

  // How to use gradient to implement a differentiable autograd function:
  // When running f:
  //   - Unwrap input Variables
  //   - Run f's graph
  //   - Create grad_fn
  //   - Wrap outputs in Variables (assume we have a tensor_outputs array):
  //       outputs = map(Variable, tensor_output)
  //       for i, offset in enumerate(df_input_vjps):
  //         outputs[offset].set_grad_fn(grad_fn, output_nr=i)
  //   - Use df_output_vjps to connect next_edges of grad_fn:
  //       for idx in df_output_vjps:
  //         grad_fn.add_next_edge(inputs[idx].gradient_edge())
  //   - Save captures for df (care needs to be taken to use SavedVariables for
  //                           inputs and outputs that we will actually return)
  //   - Return outputs[:f_real_outputs]
  //
  // When running df:
  //   - Concatenate received vjps and captured Variables
  //   - Interpret df
  //   - Wrap outputs of df into Variables (that don't require grad)
};
TORCH_API Gradient differentiate(std::shared_ptr<Graph>& graph);

// can we take a derivative of this node symbolically?
TORCH_API bool isDifferentiable(const Node* n);
TORCH_API bool isDifferentiable(Graph& g);
TORCH_API bool isZero(Value* v);

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Gradient`, `outputs`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/csrc/jit/ir/ir.h`
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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `autodiff.h_docs.md`
- **Keyword Index**: `autodiff.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/runtime`):

- [`register_ops_utils.h_docs.md_docs.md`](./register_ops_utils.h_docs.md_docs.md)
- [`register_c10_ops.cpp_docs.md_docs.md`](./register_c10_ops.cpp_docs.md_docs.md)
- [`exception_message.h_kw.md_docs.md`](./exception_message.h_kw.md_docs.md)
- [`register_prim_ops.cpp_kw.md_docs.md`](./register_prim_ops.cpp_kw.md_docs.md)
- [`autodiff.cpp_kw.md_docs.md`](./autodiff.cpp_kw.md_docs.md)
- [`decomposition_registry_util.h_docs.md_docs.md`](./decomposition_registry_util.h_docs.md_docs.md)
- [`slice_indices_adjust.cpp_docs.md_docs.md`](./slice_indices_adjust.cpp_docs.md_docs.md)
- [`graph_iterator.h_kw.md_docs.md`](./graph_iterator.h_kw.md_docs.md)
- [`shape_function_registry.h_docs.md_docs.md`](./shape_function_registry.h_docs.md_docs.md)
- [`symbolic_script.cpp_docs.md_docs.md`](./symbolic_script.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `autodiff.h_docs.md_docs.md`
- **Keyword Index**: `autodiff.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
