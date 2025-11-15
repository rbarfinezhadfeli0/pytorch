# Documentation: `docs/torch/csrc/api/include/torch/nn/modules/adaptive.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/modules/adaptive.h_docs.md`
- **Size**: 6,125 bytes (5.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/modules/adaptive.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/adaptive.h`
- **Size**: 3,517 bytes (3.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/adaptive.h>

#include <utility>

namespace torch::nn {

/// The output of a single invocation of an AdaptiveLogSoftmaxWithLoss
/// module's `forward()` method.
struct TORCH_API ASMoutput {
  ASMoutput(Tensor output_, double loss_);

  /// Tensor containing computed target log probabilities for each example
  Tensor output;

  /// Scalar representing the computed negative log likelihood loss
  double loss;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveLogSoftmaxWithLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Efficient softmax approximation as described in
/// `Efficient softmax approximation for GPUs`_ by Edouard Grave, Armand Joulin,
/// Moustapha Cissé, David Grangier, and Hervé Jégou.
/// See
/// https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveLogSoftmaxWithLoss
/// to learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::AdaptiveLogSoftmaxWithLossOptions`
/// class to learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
/// {4, 8}).div_value(2.).head_bias(true));
/// ```
class TORCH_API AdaptiveLogSoftmaxWithLossImpl
    : public Cloneable<AdaptiveLogSoftmaxWithLossImpl> {
 public:
  AdaptiveLogSoftmaxWithLossImpl(
      int64_t in_features,
      int64_t n_classes,
      std::vector<int64_t> cutoffs)
      : AdaptiveLogSoftmaxWithLossImpl(AdaptiveLogSoftmaxWithLossOptions(
            in_features,
            n_classes,
            std::move(cutoffs))) {}

  explicit AdaptiveLogSoftmaxWithLossImpl(
      AdaptiveLogSoftmaxWithLossOptions options_);

  ASMoutput forward(const Tensor& input, const Tensor& target);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `AdaptiveLogSoftmaxWithLoss` module into the given
  /// `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Given input tensor, and output of `head`, computes the log of the full
  /// distribution
  Tensor _get_full_log_prob(const Tensor& input, const Tensor& head_output);

  /// Computes log probabilities for all n_classes
  Tensor log_prob(const Tensor& input);

  /// This is equivalent to `log_pob(input).argmax(1)` but is more efficient in
  /// some cases
  Tensor predict(const Tensor& input);

  /// The options with which this `Module` was constructed
  AdaptiveLogSoftmaxWithLossOptions options;

  /// Cutoffs used to assign targets to their buckets. It should be an ordered
  /// Sequence of integers sorted in the increasing order
  std::vector<int64_t> cutoffs;

  int64_t shortlist_size;

  /// Number of clusters
  int64_t n_clusters;

  /// Output size of head classifier
  int64_t head_size;

  Linear head = nullptr;

  ModuleList tail;
};

/// A `ModuleHolder` subclass for `AdaptiveLogSoftmaxWithLossImpl`.
/// See the documentation for `AdaptiveLogSoftmaxWithLossImpl` class to learn
/// what methods it provides, and examples of how to use
/// `AdaptiveLogSoftmaxWithLoss` with
/// `torch::nn::AdaptiveLogSoftmaxWithLossOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(AdaptiveLogSoftmaxWithLoss);

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `to`, `TORCH_API`, `for`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/cloneable.h`
- `torch/nn/functional/activation.h`
- `torch/nn/module.h`
- `torch/nn/modules/container/modulelist.h`
- `torch/nn/modules/container/sequential.h`
- `torch/nn/modules/linear.h`
- `torch/nn/options/adaptive.h`
- `utility`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)


## Cross-References

- **File Documentation**: `adaptive.h_docs.md`
- **Keyword Index**: `adaptive.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/api/include/torch/nn/modules`):

- [`transformerlayer.h_docs.md_docs.md`](./transformerlayer.h_docs.md_docs.md)
- [`linear.h_kw.md_docs.md`](./linear.h_kw.md_docs.md)
- [`transformercoder.h_docs.md_docs.md`](./transformercoder.h_docs.md_docs.md)
- [`loss.h_kw.md_docs.md`](./loss.h_kw.md_docs.md)
- [`batchnorm.h_docs.md_docs.md`](./batchnorm.h_docs.md_docs.md)
- [`normalization.h_kw.md_docs.md`](./normalization.h_kw.md_docs.md)
- [`fold.h_docs.md_docs.md`](./fold.h_docs.md_docs.md)
- [`distance.h_kw.md_docs.md`](./distance.h_kw.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`transformercoder.h_kw.md_docs.md`](./transformercoder.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `adaptive.h_docs.md_docs.md`
- **Keyword Index**: `adaptive.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
