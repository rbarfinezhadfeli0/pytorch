# Documentation: `docs/torch/csrc/api/include/torch/nn/options/normalization.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/options/normalization.h_docs.md`
- **Size**: 7,905 bytes (7.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/options/normalization.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/options/normalization.h`
- **Size**: 5,477 bytes (5.35 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>
#include <vector>

namespace torch::nn {

/// Options for the `LayerNorm` module.
///
/// Example:
/// ```
/// LayerNorm model(LayerNormOptions({2,
/// 2}).elementwise_affine(false).eps(2e-5));
/// ```
struct TORCH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);
  /// a value added to the denominator for numerical stability. ``Default:
  /// 1e-5``.
  TORCH_ARG(double, eps) = 1e-5;
  /// a boolean value that when set to ``true``, this module
  /// has learnable per-element affine parameters initialized to ones (for
  /// weights) and zeros (for biases). ``Default: true``.
  TORCH_ARG(bool, elementwise_affine) = true;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::layer_norm`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
/// ```
struct TORCH_API LayerNormFuncOptions {
  /* implicit */ LayerNormFuncOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);

  TORCH_ARG(Tensor, weight);

  TORCH_ARG(Tensor, bias);

  /// a value added to the denominator for numerical stability. ``Default:
  /// 1e-5``.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

// ============================================================================

/// Options for the `LocalResponseNorm` module.
///
/// Example:
/// ```
/// LocalResponseNorm
/// model(LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.));
/// ```
struct TORCH_API LocalResponseNormOptions {
  /* implicit */ LocalResponseNormOptions(int64_t size) : size_(size) {}
  /// amount of neighbouring channels used for normalization
  TORCH_ARG(int64_t, size);

  /// multiplicative factor. Default: 1e-4
  TORCH_ARG(double, alpha) = 1e-4;

  /// exponent. Default: 0.75
  TORCH_ARG(double, beta) = 0.75;

  /// additive factor. Default: 1
  TORCH_ARG(double, k) = 1.;
};

namespace functional {
/// Options for `torch::nn::functional::local_response_norm`.
///
/// See the documentation for `torch::nn::LocalResponseNormOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
/// ```
using LocalResponseNormFuncOptions = LocalResponseNormOptions;
} // namespace functional

// ============================================================================

/// Options for the `CrossMapLRN2d` module.
///
/// Example:
/// ```
/// CrossMapLRN2d model(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10));
/// ```
struct TORCH_API CrossMapLRN2dOptions {
  CrossMapLRN2dOptions(int64_t size);

  TORCH_ARG(int64_t, size);

  TORCH_ARG(double, alpha) = 1e-4;

  TORCH_ARG(double, beta) = 0.75;

  TORCH_ARG(int64_t, k) = 1;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::normalize`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
/// ```
struct TORCH_API NormalizeFuncOptions {
  /// The exponent value in the norm formulation. Default: 2.0
  TORCH_ARG(double, p) = 2.0;
  /// The dimension to reduce. Default: 1
  TORCH_ARG(int64_t, dim) = 1;
  /// Small value to avoid division by zero. Default: 1e-12
  TORCH_ARG(double, eps) = 1e-12;
  /// the output tensor. If `out` is used, this
  /// operation won't be differentiable.
  TORCH_ARG(std::optional<Tensor>, out) = std::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `GroupNorm` module.
///
/// Example:
/// ```
/// GroupNorm model(GroupNormOptions(2, 2).eps(2e-5).affine(false));
/// ```
struct TORCH_API GroupNormOptions {
  /* implicit */ GroupNormOptions(int64_t num_groups, int64_t num_channels);

  /// number of groups to separate the channels into
  TORCH_ARG(int64_t, num_groups);
  /// number of channels expected in input
  TORCH_ARG(int64_t, num_channels);
  /// a value added to the denominator for numerical stability. Default: 1e-5
  TORCH_ARG(double, eps) = 1e-5;
  /// a boolean value that when set to ``true``, this module
  /// has learnable per-channel affine parameters initialized to ones (for
  /// weights) and zeros (for biases). Default: ``true``.
  TORCH_ARG(bool, affine) = true;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::group_norm`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
/// ```
struct TORCH_API GroupNormFuncOptions {
  /* implicit */ GroupNormFuncOptions(int64_t num_groups);

  /// number of groups to separate the channels into
  TORCH_ARG(int64_t, num_groups);

  TORCH_ARG(Tensor, weight);

  TORCH_ARG(Tensor, bias);

  /// a value added to the denominator for numerical stability. Default: 1e-5
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `functional`, `F`

**Classes/Structs**: `TORCH_API`, `TORCH_API`, `TORCH_API`, `to`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/options`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/arg.h`
- `torch/csrc/Export.h`
- `torch/types.h`
- `vector`


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

Files in the same folder (`torch/csrc/api/include/torch/nn/options`):

- [`embedding.h_docs.md`](./embedding.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`transformer.h_docs.md`](./transformer.h_docs.md)


## Cross-References

- **File Documentation**: `normalization.h_docs.md`
- **Keyword Index**: `normalization.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn/options`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn/options`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/api/include/torch/nn/options`):

- [`transformerlayer.h_docs.md_docs.md`](./transformerlayer.h_docs.md_docs.md)
- [`linear.h_kw.md_docs.md`](./linear.h_kw.md_docs.md)
- [`transformercoder.h_docs.md_docs.md`](./transformercoder.h_docs.md_docs.md)
- [`loss.h_kw.md_docs.md`](./loss.h_kw.md_docs.md)
- [`batchnorm.h_docs.md_docs.md`](./batchnorm.h_docs.md_docs.md)
- [`normalization.h_kw.md_docs.md`](./normalization.h_kw.md_docs.md)
- [`fold.h_docs.md_docs.md`](./fold.h_docs.md_docs.md)
- [`distance.h_kw.md_docs.md`](./distance.h_kw.md_docs.md)
- [`transformercoder.h_kw.md_docs.md`](./transformercoder.h_kw.md_docs.md)
- [`instancenorm.h_docs.md_docs.md`](./instancenorm.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `normalization.h_docs.md_docs.md`
- **Keyword Index**: `normalization.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
