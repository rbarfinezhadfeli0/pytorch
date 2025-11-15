# Documentation: `torch/csrc/autograd/autograd.h`

## File Metadata

- **Path**: `torch/csrc/autograd/autograd.h`
- **Size**: 5,309 bytes (5.18 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/autograd/variable.h>

namespace torch::autograd {

/// Computes the sum of gradients of given tensors with respect to graph leaves.
///
/// The graph is differentiated using the chain rule. If any of ``tensors``
/// are non-scalar (i.e. their data has more than one element) and require
/// gradient, then the Jacobian-vector product would be computed, in this case
/// the function additionally requires specifying `grad_tensors`. It should be a
/// sequence of matching length, that contains the "vector" in the
/// Jacobian-vector product, usually the gradient of the differentiated function
/// w.r.t. corresponding tensors
/// (`torch::Tensor()` is an acceptable value for all tensors that don't need
/// gradient tensors).
///
/// This function accumulates gradients in the leaves - you might need to zero
/// them before calling it.
///
/// \param tensors Tensors of which the derivative will be computed.
/// \param grad_tensors The "vector" in the Jacobian-vector product, usually
/// gradients
///     w.r.t. each element of corresponding tensors. `torch::Tensor()` values
///     can be specified for scalar Tensors or ones that don't require grad. If
///     a `torch::Tensor()` value would be acceptable for all grad_tensors, then
///     this argument is optional.
/// \param retain_graph If `false`, the graph used to compute the grad will be
/// freed.
///     Note that in nearly all cases setting this option to `true` is not
///     needed and often can be worked around in a much more efficient way.
///     Defaults to the value of `create_graph`.
/// \param create_graph If `true`, graph of the derivative will be constructed,
/// allowing
///     to compute higher order derivative products. Defaults to `false`.
/// \param inputs Inputs w.r.t. which the gradient will be accumulated into
///     `at::Tensor::grad`. All other Tensors will be ignored. If not provided,
///     the gradient is accumulated into all the leaf Tensors that were used to
///     compute param `tensors`.
//      When inputs are provided and a given input is not a leaf,
//      the current implementation will call its grad_fn (even though it is not
//      strictly needed to get this gradients). It is an implementation detail
//      on which the user should not rely. See
//      https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for
//      more details.
TORCH_API void backward(
    const variable_list& tensors,
    const variable_list& grad_tensors = {},
    std::optional<bool> retain_graph = std::nullopt,
    bool create_graph = false,
    const variable_list& inputs = {});

/// Computes and returns the sum of gradients of outputs with respect to the
/// inputs.
///
/// ``grad_outputs`` should be a sequence of length matching ``output``
/// containing the "vector" in Jacobian-vector product, usually the pre-computed
/// gradients w.r.t. each of the outputs. If an output doesn't require_grad,
/// then the gradient can be ``torch::Tensor()``).
///
/// \param outputs outputs of the differentiated function.
/// \param inputs Inputs w.r.t. which the gradient will be
///     returned (and not accumulated into ``at::Tensor::grad``).
/// \param grad_outputs The "vector" in the Jacobian-vector product.
///     Usually gradients w.r.t. each output. `torch::Tensor()` values can be
///     specified for scalar Tensors or ones that don't require grad. If a
///     `torch::Tensor()` value would be acceptable for all grad_tensors, then
///     this argument is optional. Default: `{}`.
/// \param retain_graph If ``false``, the graph used to compute the grad
///     will be freed. Note that in nearly all cases setting this option to
///     ``true`` is not needed and often can be worked around in a much more
///     efficient way. Defaults to the value of ``create_graph``.
/// \param create_graph If ``true``, graph of the derivative will
///     be constructed, allowing to compute higher order derivative products.
///     Default: ``false``.
/// \param allow_unused If ``false``, specifying inputs that were not
///     used when computing outputs (and therefore their grad is always zero)
///     is an error. Defaults to ``false``.
TORCH_API variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs = {},
    std::optional<bool> retain_graph = std::nullopt,
    bool create_graph = false,
    bool allow_unused = false);

namespace forward_ad {

/// Creates a new dual level and returns its index. This level index should then
/// be used to call into the other functions below. This API supports entering a
/// new level before the previous one is exited. We call them nested forward AD
/// levels. These can be used to compute higher order derivatives.
TORCH_API uint64_t enter_dual_level();

/// Exits the given level. This will clear up all the gradients from this level
/// and all dual Tensors that had gradients for this level will become regular
/// Tensors again. This function can only be used to exit the innermost nesting
/// level and so exiting must happen in reverse order compared to the entering
/// that was done with the function above.
TORCH_API void exit_dual_level(uint64_t level);

} // namespace forward_ad
} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `forward_ad`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/autograd/variable.h`


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

- **File Documentation**: `autograd.h_docs.md`
- **Keyword Index**: `autograd.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
