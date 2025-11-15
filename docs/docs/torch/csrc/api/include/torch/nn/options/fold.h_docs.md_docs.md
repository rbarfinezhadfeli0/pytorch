# Documentation: `docs/torch/csrc/api/include/torch/nn/options/fold.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/options/fold.h_docs.md`
- **Size**: 5,319 bytes (5.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/options/fold.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/options/fold.h`
- **Size**: 2,913 bytes (2.84 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch::nn {

/// Options for the `Fold` module.
///
/// Example:
/// ```
/// Fold model(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2,
/// 1}).stride(2));
/// ```
struct TORCH_API FoldOptions {
  FoldOptions(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : output_size_(output_size), kernel_size_(kernel_size) {}

  /// describes the spatial shape of the large containing tensor of the sliding
  /// local blocks. It is useful to resolve the ambiguity when multiple input
  /// shapes map to same number of sliding blocks, e.g., with stride > 0.
  TORCH_ARG(ExpandingArray<2>, output_size);

  /// the size of the sliding blocks
  TORCH_ARG(ExpandingArray<2>, kernel_size);

  /// controls the spacing between the kernel points; also known as the à trous
  /// algorithm.
  TORCH_ARG(ExpandingArray<2>, dilation) = 1;

  /// controls the amount of implicit zero-paddings on both sides for padding
  /// number of points for each dimension before reshaping.
  TORCH_ARG(ExpandingArray<2>, padding) = 0;

  /// controls the stride for the sliding blocks.
  TORCH_ARG(ExpandingArray<2>, stride) = 1;
};

namespace functional {
/// Options for `torch::nn::functional::fold`.
///
/// See the documentation for `torch::nn::FoldOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
/// ```
using FoldFuncOptions = FoldOptions;
} // namespace functional

// ============================================================================

/// Options for the `Unfold` module.
///
/// Example:
/// ```
/// Unfold model(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2));
/// ```
struct TORCH_API UnfoldOptions {
  UnfoldOptions(ExpandingArray<2> kernel_size) : kernel_size_(kernel_size) {}

  /// the size of the sliding blocks
  TORCH_ARG(ExpandingArray<2>, kernel_size);

  /// controls the spacing between the kernel points; also known as the à trous
  /// algorithm.
  TORCH_ARG(ExpandingArray<2>, dilation) = 1;

  /// controls the amount of implicit zero-paddings on both sides for padding
  /// number of points for each dimension before reshaping.
  TORCH_ARG(ExpandingArray<2>, padding) = 0;

  /// controls the stride for the sliding blocks.
  TORCH_ARG(ExpandingArray<2>, stride) = 1;
};

namespace functional {
/// Options for `torch::nn::functional::unfold`.
///
/// See the documentation for `torch::nn::UnfoldOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
/// ```
using UnfoldFuncOptions = UnfoldOptions;
} // namespace functional

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `functional`, `F`

**Classes/Structs**: `TORCH_API`, `to`, `TORCH_API`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/options`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/arg.h`
- `torch/csrc/Export.h`
- `torch/expanding_array.h`
- `torch/types.h`


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
- [`normalization.h_docs.md`](./normalization.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`transformer.h_docs.md`](./transformer.h_docs.md)


## Cross-References

- **File Documentation**: `fold.h_docs.md`
- **Keyword Index**: `fold.h_kw.md`
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
- [`distance.h_kw.md_docs.md`](./distance.h_kw.md_docs.md)
- [`transformercoder.h_kw.md_docs.md`](./transformercoder.h_kw.md_docs.md)
- [`instancenorm.h_docs.md_docs.md`](./instancenorm.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `fold.h_docs.md_docs.md`
- **Keyword Index**: `fold.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
