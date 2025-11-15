# Documentation: `torch/csrc/api/include/torch/nn/functional/dropout.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/functional/dropout.h`
- **Size**: 6,536 bytes (6.38 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/options/dropout.h>

#include <utility>

namespace torch::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor dropout(Tensor input, double p, bool training, bool inplace) {
  TORCH_CHECK(
      p >= 0. && p <= 1.,
      "dropout probability has to be between 0 and 1, but got ",
      p);
  if (inplace) {
    return torch::dropout_(input, p, training);
  } else {
    return torch::dropout(input, p, training);
  }
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.dropout
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::DropoutFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout(input, F::DropoutFuncOptions().p(0.5));
/// ```
inline Tensor dropout(Tensor input, const DropoutFuncOptions& options = {}) {
  return detail::dropout(
      std::move(input), options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

template <int64_t unbatched_dim, int64_t batched_dim>
inline Tensor _dropoutNd_helper(
    Tensor input,
    double p,
    bool training,
    bool inplace,
    const char* fn_name) {
  TORCH_CHECK(
      p >= 0. && p <= 1.,
      "dropout probability has to be between 0 and 1, but got ",
      p);

  auto inp_dim = input.dim();
  auto is_batched = inp_dim == batched_dim;
  if (!is_batched) {
    if (inplace) {
      input = input.unsqueeze_(0);
    } else {
      input = input.unsqueeze(0);
    }
  }

  Tensor result;
  if (inplace) {
    result = torch::feature_dropout_(input, p, training);
  } else {
    result = torch::feature_dropout(input, p, training);
  }

  if (!is_batched) {
    if (inplace) {
      result = result.squeeze_(0);
    } else {
      result = result.squeeze(0);
    }
  }
  return result;
}

inline Tensor dropout2d(Tensor input, double p, bool training, bool inplace) {
  return _dropoutNd_helper<3, 4>(
      std::move(input), p, training, inplace, "dropout2d");
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.dropout2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Dropout2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout2d(input, F::Dropout2dFuncOptions().p(0.5));
/// ```
inline Tensor dropout2d(
    Tensor input,
    const Dropout2dFuncOptions& options = {}) {
  return detail::dropout2d(
      std::move(input), options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor dropout3d(Tensor input, double p, bool training, bool inplace) {
  return _dropoutNd_helper<4, 5>(
      std::move(input), p, training, inplace, "dropout3d");
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.dropout3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Dropout3dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout3d(input, F::Dropout3dFuncOptions().p(0.5));
/// ```
inline Tensor dropout3d(
    Tensor input,
    const Dropout3dFuncOptions& options = {}) {
  return detail::dropout3d(
      std::move(input), options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor alpha_dropout(
    Tensor input,
    double p,
    bool training,
    bool inplace) {
  if (p < 0. || p > 1.) {
    TORCH_CHECK(
        false, "dropout probability has to be between 0 and 1, but got ", p);
  }
  return inplace ? torch::alpha_dropout_(input, p, training)
                 : torch::alpha_dropout(input, p, training);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.alpha_dropout
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::AlphaDropoutFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::alpha_dropout(input,
/// F::AlphaDropoutFuncOptions().p(0.5).training(false));
/// ```
inline Tensor alpha_dropout(
    Tensor input,
    const AlphaDropoutFuncOptions& options = {}) {
  return detail::alpha_dropout(
      std::move(input), options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor feature_alpha_dropout(
    Tensor input,
    double p,
    bool training,
    bool inplace) {
  if (p < 0. || p > 1.) {
    TORCH_CHECK(
        false, "dropout probability has to be between 0 and 1, but got ", p);
  }
  return inplace ? torch::feature_alpha_dropout_(input, p, training)
                 : torch::feature_alpha_dropout(input, p, training);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.feature_alpha_dropout
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::FeatureAlphaDropoutFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::feature_alpha_dropout(input,
/// F::FeatureAlphaDropoutFuncOptions().p(0.5).training(false));
/// ```
inline Tensor feature_alpha_dropout(
    Tensor input,
    const FeatureAlphaDropoutFuncOptions& options = {}) {
  return detail::feature_alpha_dropout(
      std::move(input), options.p(), options.training(), options.inplace());
}

} // namespace torch::nn::functional

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`, `F`

**Classes/Structs**: `to`, `to`, `to`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/functional`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/options/dropout.h`
- `utility`


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

- **File Documentation**: `dropout.h_docs.md`
- **Keyword Index**: `dropout.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
