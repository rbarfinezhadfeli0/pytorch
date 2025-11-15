# Documentation: `torch/csrc/api/include/torch/nn/modules/embedding.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/embedding.h`
- **Size**: 6,055 bytes (5.91 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/embedding.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/options/embedding.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>

namespace torch::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Embedding
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Performs a lookup in a fixed size embedding table.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Embedding to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::EmbeddingOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Embedding model(EmbeddingOptions(10,
/// 2).padding_idx(3).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
class TORCH_API EmbeddingImpl : public torch::nn::Cloneable<EmbeddingImpl> {
 public:
  EmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim)
      : EmbeddingImpl(EmbeddingOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingImpl(EmbeddingOptions options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `Embedding` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Performs a lookup on the embedding table stored in `weight` using the
  /// `indices` supplied and returns the result.
  Tensor forward(const Tensor& indices);

  /// The `Options` used to configure this `Embedding` module.
  /// Changes to `EmbeddingOptions` *after construction* have no effect.
  EmbeddingOptions options;

  /// The embedding table.
  Tensor weight;
};

/// A `ModuleHolder` subclass for `EmbeddingImpl`.
/// See the documentation for `EmbeddingImpl` class to learn what methods it
/// provides, and examples of how to use `Embedding` with
/// `torch::nn::EmbeddingOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
class Embedding : public torch::nn::ModuleHolder<EmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<EmbeddingImpl>::ModuleHolder;

  /// See the documentation for `torch::nn::EmbeddingFromPretrainedOptions`
  /// class to learn what optional arguments are supported for this function.
  static Embedding from_pretrained(
      const torch::Tensor& embeddings,
      const EmbeddingFromPretrainedOptions& options = {}) {
    TORCH_CHECK(
        embeddings.dim() == 2,
        "Embeddings parameter is expected to be 2-dimensional");

    auto rows = embeddings.size(0);
    auto cols = embeddings.size(1);

    Embedding embedding(EmbeddingOptions(rows, cols)
                            ._weight(embeddings)
                            .padding_idx(options.padding_idx())
                            .max_norm(options.max_norm())
                            .norm_type(options.norm_type())
                            .scale_grad_by_freq(options.scale_grad_by_freq())
                            .sparse(options.sparse()));
    embedding->weight.set_requires_grad(!options.freeze());
    return embedding;
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EmbeddingBag
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Computes sums or means of 'bags' of embeddings, without instantiating the
/// intermediate embeddings.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.EmbeddingBag to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::EmbeddingBagOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// EmbeddingBag model(EmbeddingBagOptions(10,
/// 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch::kSum).padding_idx(1));
/// ```
class TORCH_API EmbeddingBagImpl
    : public torch::nn::Cloneable<EmbeddingBagImpl> {
 public:
  EmbeddingBagImpl(int64_t num_embeddings, int64_t embedding_dim)
      : EmbeddingBagImpl(EmbeddingBagOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingBagImpl(EmbeddingBagOptions options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `EmbeddingBag` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The `Options` used to configure this `EmbeddingBag` module.
  EmbeddingBagOptions options;
  /// The embedding table.
  Tensor weight;

  Tensor forward(
      const Tensor& input,
      const Tensor& offsets = {},
      const Tensor& per_sample_weights = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})
};

/// A `ModuleHolder` subclass for `EmbeddingBagImpl`.
/// See the documentation for `EmbeddingBagImpl` class to learn what methods it
/// provides, and examples of how to use `EmbeddingBag` with
/// `torch::nn::EmbeddingBagOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
class EmbeddingBag : public torch::nn::ModuleHolder<EmbeddingBagImpl> {
 public:
  using torch::nn::ModuleHolder<EmbeddingBagImpl>::ModuleHolder;

  /// See the documentation for `torch::nn::EmbeddingBagFromPretrainedOptions`
  /// class to learn what optional arguments are supported for this function.
  static EmbeddingBag from_pretrained(
      const torch::Tensor& embeddings,
      const EmbeddingBagFromPretrainedOptions& options = {}) {
    TORCH_CHECK(
        embeddings.dim() == 2,
        "Embeddings parameter is expected to be 2-dimensional");

    auto rows = embeddings.size(0);
    auto cols = embeddings.size(1);

    EmbeddingBag embeddingbag(
        EmbeddingBagOptions(rows, cols)
            ._weight(embeddings)
            .max_norm(options.max_norm())
            .norm_type(options.norm_type())
            .scale_grad_by_freq(options.scale_grad_by_freq())
            .mode(options.mode())
            .sparse(options.sparse())
            .padding_idx(options.padding_idx()));
    embeddingbag->weight.set_requires_grad(!options.freeze());
    return embeddingbag;
  }
};
} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 12 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `to`, `TORCH_API`, `for`, `to`, `Embedding`, `to`, `to`, `TORCH_API`, `for`, `to`, `EmbeddingBag`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/cloneable.h`
- `torch/nn/functional/embedding.h`
- `torch/nn/modules/common.h`
- `torch/nn/options/embedding.h`
- `torch/nn/pimpl.h`
- `torch/types.h`
- `cstddef`


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

- **File Documentation**: `embedding.h_docs.md`
- **Keyword Index**: `embedding.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
