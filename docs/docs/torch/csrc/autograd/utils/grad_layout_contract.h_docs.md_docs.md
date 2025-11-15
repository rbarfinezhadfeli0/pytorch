# Documentation: `docs/torch/csrc/autograd/utils/grad_layout_contract.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/utils/grad_layout_contract.h_docs.md`
- **Size**: 4,952 bytes (4.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/utils/grad_layout_contract.h`

## File Metadata

- **Path**: `torch/csrc/autograd/utils/grad_layout_contract.h`
- **Size**: 2,822 bytes (2.76 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Tensor.h>

namespace torch::autograd::utils {

// Helper functions to enforce the "Gradient Layout Contract" described in
// torch/csrc/autograd/functions/accumulate_grad.h.

// Checks if grad obeys the contract with variable.
inline bool obeys_layout_contract(
    const at::Tensor& grad,
    const at::Tensor& variable) {
  TORCH_INTERNAL_ASSERT(!grad.is_sparse());
  TORCH_INTERNAL_ASSERT(!grad.is_sparse_csr());
  TORCH_INTERNAL_ASSERT(!variable.is_sparse_csr());

  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (variable.is_nested()) {
    // TODO: Nested Tensor does not have an implementation of detach. The
    // current implementation of nested tensor likely does obey the gradient
    // contract and should return true, but this would likely change in the
    // future
    return false;
  } else if (variable.is_sparse()) {
    // Gradient Layout Contract is not applicable for sparse layouts
    return false;
  } else if (variable.is_non_overlapping_and_dense()) {
    // Only look at stride for dimensions that are not of size 1.
    const auto& grad_sizes = grad.sym_sizes();
    const auto& grad_strides = grad.sym_strides();
    const auto& variable_strides = variable.sym_strides();
    for (const auto idx : c10::irange(grad_sizes.size())) {
      if (grad_sizes[idx] != 1) {
        if (grad_strides[idx] != variable_strides[idx]) {
          return false;
        }
      } else {
        // This should not be needed but we don't check if a Tensor has views
        // before stashing it. And 0-strided Tensors of size 1 are actually
        // views for ops like cat.
        // TODO: Actually detect views in the accumulateGrad function so that
        // this Tensor is not considered at all.
        if (grad_strides[idx] == 0) {
          return false;
        }
      }
    }
    return true;
  } else {
    return grad.is_contiguous(at::MemoryFormat::Contiguous);
  }
}

// Creates a clone of new_grad that obeys the contract with variable.
// The clone should attach to new_grad's history if GradMode::is_enabled().
inline at::Tensor clone_obey_contract(
    const at::Tensor& new_grad,
    const at::Tensor& variable) {
  if (variable.is_non_overlapping_and_dense()) {
    // (1)
    // Does this dicey-looking sequence attach the result to new_grad's
    // history if GradMode::is_enabled()?  Yes, and @alband says it should.
    return std::move(new_grad
                         .new_empty_strided_symint(
                             variable.sym_sizes(),
                             variable.sym_strides(),
                             variable.options().memory_format(std::nullopt))
                         .copy_(new_grad));
  } else {
    // (2)
    return new_grad.clone(at::MemoryFormat::Contiguous);
  }
}

} // namespace torch::autograd::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`


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
- [`wrap_outputs.h_docs.md`](./wrap_outputs.h_docs.md)
- [`lambda_post_hook.h_docs.md`](./lambda_post_hook.h_docs.md)
- [`python_arg_parsing.h_docs.md`](./python_arg_parsing.h_docs.md)


## Cross-References

- **File Documentation**: `grad_layout_contract.h_docs.md`
- **Keyword Index**: `grad_layout_contract.h_kw.md`
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

- [`error_messages.h_kw.md_docs.md`](./error_messages.h_kw.md_docs.md)
- [`error_messages.h_docs.md_docs.md`](./error_messages.h_docs.md_docs.md)
- [`lambda_post_hook.h_kw.md_docs.md`](./lambda_post_hook.h_kw.md_docs.md)
- [`warnings.cpp_kw.md_docs.md`](./warnings.cpp_kw.md_docs.md)
- [`python_arg_parsing.h_docs.md_docs.md`](./python_arg_parsing.h_docs.md_docs.md)
- [`wrap_outputs.h_docs.md_docs.md`](./wrap_outputs.h_docs.md_docs.md)
- [`grad_layout_contract.h_kw.md_docs.md`](./grad_layout_contract.h_kw.md_docs.md)
- [`python_arg_parsing.h_kw.md_docs.md`](./python_arg_parsing.h_kw.md_docs.md)
- [`warnings.cpp_docs.md_docs.md`](./warnings.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `grad_layout_contract.h_docs.md_docs.md`
- **Keyword Index**: `grad_layout_contract.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
