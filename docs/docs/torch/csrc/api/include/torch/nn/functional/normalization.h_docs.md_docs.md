# Documentation: `docs/torch/csrc/api/include/torch/nn/functional/normalization.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/functional/normalization.h_docs.md`
- **Size**: 8,394 bytes (8.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/functional/normalization.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/functional/normalization.h`
- **Size**: 5,971 bytes (5.83 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/functional/padding.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/options/normalization.h>
#include <torch/types.h>

namespace torch::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor normalize(
    const Tensor& input,
    double p,
    int64_t dim,
    double eps,
    std::optional<Tensor> out) {
  if (out == std::nullopt) {
    auto denom = input.norm(p, dim, true).clamp_min(eps).expand_as(input);
    return input / denom;
  } else {
    auto denom = input.norm(p, dim, true).clamp_min(eps).expand_as(input);
    return torch::div_out(*out, input, denom);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.normalize
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::NormalizeFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
/// ```
inline Tensor normalize(
    const Tensor& input,
    NormalizeFuncOptions options = {}) {
  return detail::normalize(
      input, options.p(), options.dim(), options.eps(), options.out());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor layer_norm(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  return torch::layer_norm(input, normalized_shape, weight, bias, eps);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.layer_norm
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::LayerNormFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
/// ```
inline Tensor layer_norm(
    const Tensor& input,
    const LayerNormFuncOptions& options) {
  return detail::layer_norm(
      input,
      options.normalized_shape(),
      options.weight(),
      options.bias(),
      options.eps());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor local_response_norm(
    const Tensor& input,
    int64_t size,
    double alpha,
    double beta,
    double k) {
  auto dim = input.dim();
  TORCH_CHECK(
      dim >= 3,
      "Expected 3D or higher dimensionality input (got ",
      dim,
      " dimensions)");
  auto div = input.mul(input).unsqueeze(1);
  if (dim == 3) {
    div = detail::pad(
        div,
        /*pad=*/{0, 0, size / 2, (size - 1) / 2},
        /*mode=*/torch::kConstant,
        /*value=*/0);
    div = detail::avg_pool2d(
              div,
              /*kernel_size=*/{size, 1},
              /*stride=*/1,
              /*padding=*/0,
              /*ceil_mode=*/false,
              /*count_include_pad=*/true,
              /*divisor_override=*/std::nullopt)
              .squeeze(1);
  } else {
    auto sizes = input.sizes();
    div = div.view({sizes[0], 1, sizes[1], sizes[2], -1});
    div = detail::pad(
        div,
        /*pad=*/{0, 0, 0, 0, size / 2, (size - 1) / 2},
        /*mode=*/torch::kConstant,
        /*value=*/0);
    div = detail::avg_pool3d(
              div,
              /*kernel_size=*/{size, 1, 1},
              /*stride=*/1,
              /*padding=*/0,
              /*ceil_mode=*/false,
              /*count_include_pad=*/true,
              /*divisor_override=*/std::nullopt)
              .squeeze(1);
    div = div.view(sizes);
  }
  div = div.mul(alpha).add(k).pow(beta);
  return input / div;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.local_response_norm
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::LocalResponseNormFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
/// ```
inline Tensor local_response_norm(
    const Tensor& input,
    const LocalResponseNormFuncOptions& options) {
  return detail::local_response_norm(
      input, options.size(), options.alpha(), options.beta(), options.k());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor group_norm(
    const Tensor& input,
    int64_t num_groups,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  return torch::group_norm(
      input,
      num_groups,
      weight,
      bias,
      eps,
      at::globalContext().userEnabledCuDNN());
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.group_norm
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::GroupNormFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
/// ```
inline Tensor group_norm(
    const Tensor& input,
    const GroupNormFuncOptions& options) {
  return detail::group_norm(
      input,
      options.num_groups(),
      options.weight(),
      options.bias(),
      options.eps());
}

} // namespace torch::nn::functional

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 9 function(s).

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

- `torch/nn/functional/padding.h`
- `torch/nn/functional/pooling.h`
- `torch/nn/options/normalization.h`
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

Files in the same folder (`torch/csrc/api/include/torch/nn/functional`):

- [`embedding.h_docs.md`](./embedding.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`vision.h_docs.md`](./vision.h_docs.md)
- [`conv.h_docs.md`](./conv.h_docs.md)


## Cross-References

- **File Documentation**: `normalization.h_docs.md`
- **Keyword Index**: `normalization.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn/functional`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn/functional`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/api/include/torch/nn/functional`):

- [`linear.h_kw.md_docs.md`](./linear.h_kw.md_docs.md)
- [`loss.h_kw.md_docs.md`](./loss.h_kw.md_docs.md)
- [`batchnorm.h_docs.md_docs.md`](./batchnorm.h_docs.md_docs.md)
- [`normalization.h_kw.md_docs.md`](./normalization.h_kw.md_docs.md)
- [`fold.h_docs.md_docs.md`](./fold.h_docs.md_docs.md)
- [`distance.h_kw.md_docs.md`](./distance.h_kw.md_docs.md)
- [`instancenorm.h_docs.md_docs.md`](./instancenorm.h_docs.md_docs.md)
- [`dropout.h_kw.md_docs.md`](./dropout.h_kw.md_docs.md)
- [`dropout.h_docs.md_docs.md`](./dropout.h_docs.md_docs.md)
- [`fold.h_kw.md_docs.md`](./fold.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `normalization.h_docs.md_docs.md`
- **Keyword Index**: `normalization.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
