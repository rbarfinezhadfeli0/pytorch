# Documentation: `torch/csrc/api/include/torch/nn/modules/fold.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/fold.h`
- **Size**: 2,822 bytes (2.76 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/functional/fold.h>
#include <torch/nn/options/fold.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

namespace torch::nn {

/// Applies fold over a 3-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Fold to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::FoldOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Fold model(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2,
/// 1}).stride(2));
/// ```
class TORCH_API FoldImpl : public torch::nn::Cloneable<FoldImpl> {
 public:
  FoldImpl(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : FoldImpl(FoldOptions(output_size, kernel_size)) {}
  explicit FoldImpl(const FoldOptions& options_);

  void reset() override;

  /// Pretty prints the `Fold` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  FoldOptions options;
};

/// A `ModuleHolder` subclass for `FoldImpl`.
/// See the documentation for `FoldImpl` class to learn what methods it
/// provides, and examples of how to use `Fold` with `torch::nn::FoldOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Fold);

// ============================================================================

/// Applies unfold over a 4-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Unfold to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::UnfoldOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Unfold model(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2));
/// ```
class TORCH_API UnfoldImpl : public Cloneable<UnfoldImpl> {
 public:
  UnfoldImpl(ExpandingArray<2> kernel_size)
      : UnfoldImpl(UnfoldOptions(kernel_size)) {}
  explicit UnfoldImpl(const UnfoldOptions& options_);

  void reset() override;

  /// Pretty prints the `Unfold` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  UnfoldOptions options;
};

/// A `ModuleHolder` subclass for `UnfoldImpl`.
/// See the documentation for `UnfoldImpl` class to learn what methods it
/// provides, and examples of how to use `Unfold` with
/// `torch::nn::UnfoldOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Unfold);

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 8 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `to`, `TORCH_API`, `for`, `to`, `to`, `TORCH_API`, `for`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/expanding_array.h`
- `torch/nn/cloneable.h`
- `torch/nn/functional/fold.h`
- `torch/nn/options/fold.h`
- `torch/nn/pimpl.h`
- `torch/types.h`


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

Files in the same folder (`torch/csrc/api/include/torch/nn/modules`):

- [`embedding.h_docs.md`](./embedding.h_docs.md)
- [`normalization.h_docs.md`](./normalization.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`common.h_docs.md`](./common.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)


## Cross-References

- **File Documentation**: `fold.h_docs.md`
- **Keyword Index**: `fold.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
