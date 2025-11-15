# Documentation: `torch/csrc/api/include/torch/nn/functional/conv.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/functional/conv.h`
- **Size**: 8,119 bytes (7.93 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/options/conv.h>
#include <torch/types.h>

namespace torch::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline std::string padding_unwrap(enumtype::kValid /*unused*/) {
  return "valid";
}

inline std::string padding_unwrap(enumtype::kSame /*unused*/) {
  return "same";
}

template <size_t D>
IntArrayRef padding_unwrap(const ExpandingArray<D>& array) {
  return array;
}

inline Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<1> stride,
    const Conv1dFuncOptions::padding_t& padding,
    ExpandingArray<1> dilation,
    int64_t groups) {
  return std::visit(
      [&](const auto& pad) {
        return torch::conv1d(
            input, weight, bias, stride, padding_unwrap(pad), dilation, groups);
      },
      padding);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.conv1d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Conv1dFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));
/// ```
inline Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Conv1dFuncOptions& options = {}) {
  return detail::conv1d(
      input,
      weight,
      options.bias(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.groups());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<2> stride,
    const Conv2dFuncOptions::padding_t& padding,
    ExpandingArray<2> dilation,
    int64_t groups) {
  return std::visit(
      [&](const auto& pad) {
        return torch::conv2d(
            input, weight, bias, stride, padding_unwrap(pad), dilation, groups);
      },
      padding);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.conv2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Conv2dFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));
/// ```
inline Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Conv2dFuncOptions& options = {}) {
  return detail::conv2d(
      input,
      weight,
      options.bias(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.groups());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<3> stride,
    const Conv3dFuncOptions::padding_t& padding,
    ExpandingArray<3> dilation,
    int64_t groups) {
  return std::visit(
      [&](const auto& pad) {
        return torch::conv3d(
            input, weight, bias, stride, padding_unwrap(pad), dilation, groups);
      },
      padding);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.conv3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Conv3dFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv3d(x, weight, F::Conv3dFuncOptions().stride(1));
/// ```
inline Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const Conv3dFuncOptions& options = {}) {
  return detail::conv3d(
      input,
      weight,
      options.bias(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.groups());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv_transpose1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  return torch::conv_transpose1d(
      input, weight, bias, stride, padding, output_padding, groups, dilation);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.conv_transpose1d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::ConvTranspose1dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose1d(x, weight, F::ConvTranspose1dFuncOptions().stride(1));
/// ```
inline Tensor conv_transpose1d(
    const Tensor& input,
    const Tensor& weight,
    const ConvTranspose1dFuncOptions& options = {}) {
  return detail::conv_transpose1d(
      input,
      weight,
      options.bias(),
      options.stride(),
      options.padding(),
      options.output_padding(),
      options.groups(),
      options.dilation());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv_transpose2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  return torch::conv_transpose2d(
      input, weight, bias, stride, padding, output_padding, groups, dilation);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.conv_transpose2d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::ConvTranspose2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose2d(x, weight, F::ConvTranspose2dFuncOptions().stride(1));
/// ```
inline Tensor conv_transpose2d(
    const Tensor& input,
    const Tensor& weight,
    const ConvTranspose2dFuncOptions& options = {}) {
  return detail::conv_transpose2d(
      input,
      weight,
      options.bias(),
      options.stride(),
      options.padding(),
      options.output_padding(),
      options.groups(),
      options.dilation());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv_transpose3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  return torch::conv_transpose3d(
      input, weight, bias, stride, padding, output_padding, groups, dilation);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.conv_transpose3d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::ConvTranspose3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose3d(x, weight, F::ConvTranspose3dFuncOptions().stride(1));
/// ```
inline Tensor conv_transpose3d(
    const Tensor& input,
    const Tensor& weight,
    const ConvTranspose3dFuncOptions& options = {}) {
  return detail::conv_transpose3d(
      input,
      weight,
      options.bias(),
      options.stride(),
      options.padding(),
      options.output_padding(),
      options.groups(),
      options.dilation());
}

} // namespace torch::nn::functional

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`, `F`

**Classes/Structs**: `to`, `to`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/functional`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/options/conv.h`
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
- [`normalization.h_docs.md`](./normalization.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`vision.h_docs.md`](./vision.h_docs.md)


## Cross-References

- **File Documentation**: `conv.h_docs.md`
- **Keyword Index**: `conv.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
