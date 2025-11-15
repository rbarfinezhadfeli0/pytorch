# Documentation: `docs/torch/csrc/api/include/torch/nn/functional/vision.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/functional/vision.h_docs.md`
- **Size**: 5,895 bytes (5.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/functional/vision.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/functional/vision.h`
- **Size**: 3,579 bytes (3.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/options/vision.h>
#include <torch/types.h>

namespace torch::nn::functional {

inline Tensor affine_grid(
    const Tensor& theta,
    const IntArrayRef& size,
    bool align_corners = false) {
  // enforce floating point dtype on theta
  TORCH_CHECK(
      theta.is_floating_point(),
      "Expected theta to have floating point type, but got ",
      theta.dtype());

  // check that shapes and sizes match
  if (size.size() == 4) {
    TORCH_CHECK(
        theta.dim() == 3 && theta.size(-2) == 2 && theta.size(-1) == 3,
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size ",
        size,
        ". Got ",
        theta.sizes(),
        ".");
  } else if (size.size() == 5) {
    TORCH_CHECK(
        theta.dim() == 3 && theta.size(-2) == 3 && theta.size(-1) == 4,
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size ",
        size,
        ". Got ",
        theta.sizes(),
        ".");
  } else {
    TORCH_CHECK(
        false,
        "affine_grid only supports 4D and 5D sizes, ",
        "for 2D and 3D affine transforms, respectively. ",
        "Got size ",
        size);
  }

  if (*std::min_element(size.begin(), size.end()) <= 0) {
    TORCH_CHECK(false, "Expected non-zero, positive output size. Got ", size);
  }

  return torch::affine_grid_generator(theta, size, align_corners);
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor grid_sample(
    const Tensor& input,
    const Tensor& grid,
    GridSampleFuncOptions::mode_t mode,
    GridSampleFuncOptions::padding_mode_t padding_mode,
    std::optional<bool> align_corners) {
  int64_t mode_enum = 0, padding_mode_enum = 0;

  if (std::holds_alternative<enumtype::kBilinear>(mode)) {
    mode_enum = 0;
  } else if (std::holds_alternative<enumtype::kNearest>(mode)) {
    mode_enum = 1;
  } else { /// mode == 'bicubic'
    mode_enum = 2;
  }

  if (std::holds_alternative<enumtype::kZeros>(padding_mode)) {
    padding_mode_enum = 0;
  } else if (std::holds_alternative<enumtype::kBorder>(padding_mode)) {
    padding_mode_enum = 1;
  } else { /// padding_mode == 'reflection'
    padding_mode_enum = 2;
  }

  if (!align_corners.has_value()) {
    TORCH_WARN(
        "Default grid_sample and affine_grid behavior has changed ",
        "to align_corners=False since 1.3.0. Please specify ",
        "align_corners=True if the old behavior is desired. ",
        "See the documentation of grid_sample for details.");
    align_corners = false;
  }

  return torch::grid_sampler(
      input, grid, mode_enum, padding_mode_enum, align_corners.value());
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.grid_sample
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::GridSampleFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::grid_sample(input, grid,
/// F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));
/// ```
inline Tensor grid_sample(
    const Tensor& input,
    const Tensor& grid,
    const GridSampleFuncOptions& options = {}) {
  return detail::grid_sample(
      input,
      grid,
      options.mode(),
      options.padding_mode(),
      options.align_corners());
}

} // namespace torch::nn::functional

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 8 function(s).

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

- `torch/nn/options/vision.h`
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
- [`conv.h_docs.md`](./conv.h_docs.md)


## Cross-References

- **File Documentation**: `vision.h_docs.md`
- **Keyword Index**: `vision.h_kw.md`
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

- **File Documentation**: `vision.h_docs.md_docs.md`
- **Keyword Index**: `vision.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
