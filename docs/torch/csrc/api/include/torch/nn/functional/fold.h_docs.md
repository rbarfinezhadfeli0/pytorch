# Documentation: `torch/csrc/api/include/torch/nn/functional/fold.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/functional/fold.h`
- **Size**: 2,736 bytes (2.67 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/options/fold.h>

namespace torch::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor fold(
    const Tensor& input,
    ExpandingArray<2> output_size,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> dilation,
    ExpandingArray<2> padding,
    ExpandingArray<2> stride) {
  if (input.dim() == 3 || input.dim() == 2) {
    return torch::col2im(
        input, output_size, kernel_size, dilation, padding, stride);
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only unbatched (2D) or batched (3D) input Tensors are supported "
        "(got ",
        input.dim(),
        "D)");
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.fold
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::FoldFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
/// ```
inline Tensor fold(const Tensor& input, const FoldFuncOptions& options) {
  return detail::fold(
      input,
      options.output_size(),
      options.kernel_size(),
      options.dilation(),
      options.padding(),
      options.stride());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor unfold(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> dilation,
    ExpandingArray<2> padding,
    ExpandingArray<2> stride) {
  if (input.dim() == 4) {
    return torch::im2col(input, kernel_size, dilation, padding, stride);
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only 4D input Tensors are supported "
        "(got ",
        input.dim(),
        "D)");
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.unfold
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::UnfoldFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
/// ```
inline Tensor unfold(const Tensor& input, const UnfoldFuncOptions& options) {
  return detail::unfold(
      input,
      options.kernel_size(),
      options.dilation(),
      options.padding(),
      options.stride());
}

} // namespace torch::nn::functional

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`, `F`

**Classes/Structs**: `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/functional`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/options/fold.h`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/csrc/api/include/torch/nn/functional`):

- [`embedding.h_docs.md`](./embedding.h_docs.md)
- [`normalization.h_docs.md`](./normalization.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`vision.h_docs.md`](./vision.h_docs.md)
- [`conv.h_docs.md`](./conv.h_docs.md)


## Cross-References

- **File Documentation**: `fold.h_docs.md`
- **Keyword Index**: `fold.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
