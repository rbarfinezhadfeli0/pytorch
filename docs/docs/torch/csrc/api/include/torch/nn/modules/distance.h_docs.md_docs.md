# Documentation: `docs/torch/csrc/api/include/torch/nn/modules/distance.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/modules/distance.h_docs.md`
- **Size**: 5,520 bytes (5.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/modules/distance.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/distance.h`
- **Size**: 3,056 bytes (2.98 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/distance.h>
#include <torch/nn/options/distance.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/Export.h>

namespace torch::nn {

/// Returns the cosine similarity between :math:`x_1` and :math:`x_2`, computed
/// along `dim`.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.CosineSimilarity to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::CosineSimilarityOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CosineSimilarity model(CosineSimilarityOptions().dim(0).eps(0.5));
/// ```
class TORCH_API CosineSimilarityImpl : public Cloneable<CosineSimilarityImpl> {
 public:
  explicit CosineSimilarityImpl(const CosineSimilarityOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `CosineSimilarity` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// The options with which this `Module` was constructed.
  CosineSimilarityOptions options;
};

/// A `ModuleHolder` subclass for `CosineSimilarityImpl`.
/// See the documentation for `CosineSimilarityImpl` class to learn what methods
/// it provides, and examples of how to use `CosineSimilarity` with
/// `torch::nn::CosineSimilarityOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(CosineSimilarity);

// ============================================================================

/// Returns the batchwise pairwise distance between vectors :math:`v_1`,
/// :math:`v_2` using the p-norm.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.PairwiseDistance to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::PairwiseDistanceOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PairwiseDistance
/// model(PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true));
/// ```
class TORCH_API PairwiseDistanceImpl : public Cloneable<PairwiseDistanceImpl> {
 public:
  explicit PairwiseDistanceImpl(const PairwiseDistanceOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `PairwiseDistance` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// The options with which this `Module` was constructed.
  PairwiseDistanceOptions options;
};

/// A `ModuleHolder` subclass for `PairwiseDistanceImpl`.
/// See the documentation for `PairwiseDistanceImpl` class to learn what methods
/// it provides, and examples of how to use `PairwiseDistance` with
/// `torch::nn::PairwiseDistanceOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(PairwiseDistance);

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 8 class(es)/struct(s) and 9 function(s).

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
- `torch/nn/functional/distance.h`
- `torch/nn/options/distance.h`
- `torch/nn/pimpl.h`
- `torch/types.h`
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
- [`common.h_docs.md`](./common.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)


## Cross-References

- **File Documentation**: `distance.h_docs.md`
- **Keyword Index**: `distance.h_kw.md`
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

- **File Documentation**: `distance.h_docs.md_docs.md`
- **Keyword Index**: `distance.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
