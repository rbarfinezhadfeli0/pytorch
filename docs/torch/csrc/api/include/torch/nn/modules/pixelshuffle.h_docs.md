# Documentation: `torch/csrc/api/include/torch/nn/modules/pixelshuffle.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/pixelshuffle.h`
- **Size**: 3,109 bytes (3.04 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/pixelshuffle.h>
#include <torch/nn/options/pixelshuffle.h>

#include <torch/csrc/Export.h>

namespace torch::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelShuffle
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
/// to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an
/// upscale factor. See
/// https://pytorch.org/docs/main/nn.html#torch.nn.PixelShuffle to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::PixelShuffleOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelShuffle model(PixelShuffleOptions(5));
/// ```
struct TORCH_API PixelShuffleImpl
    : public torch::nn::Cloneable<PixelShuffleImpl> {
  explicit PixelShuffleImpl(const PixelShuffleOptions& options_);

  /// Pretty prints the `PixelShuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  void reset() override;

  /// The options with which this `Module` was constructed.
  PixelShuffleOptions options;
};

/// A `ModuleHolder` subclass for `PixelShuffleImpl`.
/// See the documentation for `PixelShuffleImpl` class to learn what methods it
/// provides, and examples of how to use `PixelShuffle` with
/// `torch::nn::PixelShuffleOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(PixelShuffle);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelUnshuffle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Reverses the PixelShuffle operation by rearranging elements in a tensor of
/// shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape :math:`(*,
/// C \times r^2, H, W)`, where r is a downscale factor. See
/// https://pytorch.org/docs/main/nn.html#torch.nn.PixelUnshuffle to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::PixelUnshuffleOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelUnshuffle model(PixelUnshuffleOptions(5));
/// ```
struct TORCH_API PixelUnshuffleImpl
    : public torch::nn::Cloneable<PixelUnshuffleImpl> {
  explicit PixelUnshuffleImpl(const PixelUnshuffleOptions& options_);

  /// Pretty prints the `PixelUnshuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  void reset() override;

  /// The options with which this `Module` was constructed.
  PixelUnshuffleOptions options;
};

/// A `ModuleHolder` subclass for `PixelUnshuffleImpl`.
/// See the documentation for `PixelUnshuffleImpl` class to learn what methods
/// it provides, and examples of how to use `PixelUnshuffle` with
/// `torch::nn::PixelUnshuffleOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(PixelUnshuffle);

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 6 class(es)/struct(s) and 10 function(s).

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

- `torch/nn/cloneable.h`
- `torch/nn/functional/pixelshuffle.h`
- `torch/nn/options/pixelshuffle.h`
- `torch/csrc/Export.h`


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

- **File Documentation**: `pixelshuffle.h_docs.md`
- **Keyword Index**: `pixelshuffle.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
