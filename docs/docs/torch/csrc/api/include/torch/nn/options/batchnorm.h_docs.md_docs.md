# Documentation: `docs/torch/csrc/api/include/torch/nn/options/batchnorm.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/options/batchnorm.h_docs.md`
- **Size**: 5,074 bytes (4.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/options/batchnorm.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/options/batchnorm.h`
- **Size**: 2,737 bytes (2.67 KB)
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

namespace torch::nn {

/// Options for the `BatchNorm` module.
struct TORCH_API BatchNormOptions {
  /* implicit */ BatchNormOptions(int64_t num_features);

  /// The number of features of the input tensor.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, num_features);

  /// The epsilon value added for numerical stability.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, eps) = 1e-5;

  /// A momentum multiplier for the mean and variance.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(std::optional<double>, momentum) = 0.1;

  /// Whether to learn a scale and bias that are applied in an affine
  /// transformation on the input.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, affine) = true;

  /// Whether to store and update batch statistics (mean and variance) in the
  /// module.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, track_running_stats) = true;
};

/// Options for the `BatchNorm1d` module.
///
/// Example:
/// ```
/// BatchNorm1d
/// model(BatchNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using BatchNorm1dOptions = BatchNormOptions;

/// Options for the `BatchNorm2d` module.
///
/// Example:
/// ```
/// BatchNorm2d
/// model(BatchNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using BatchNorm2dOptions = BatchNormOptions;

/// Options for the `BatchNorm3d` module.
///
/// Example:
/// ```
/// BatchNorm3d
/// model(BatchNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using BatchNorm3dOptions = BatchNormOptions;

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::batch_norm`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::batch_norm(input, mean, variance,
/// F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(0.1).eps(1e-05).training(false));
/// ```
struct TORCH_API BatchNormFuncOptions {
  TORCH_ARG(Tensor, weight);

  TORCH_ARG(Tensor, bias);

  TORCH_ARG(bool, training) = false;

  /// A momentum multiplier for the mean and variance.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, momentum) = 0.1;

  /// The epsilon value added for numerical stability.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `functional`, `F`

**Classes/Structs**: `TORCH_API`, `TORCH_API`


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
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`transformer.h_docs.md`](./transformer.h_docs.md)


## Cross-References

- **File Documentation**: `batchnorm.h_docs.md`
- **Keyword Index**: `batchnorm.h_kw.md`
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
- [`normalization.h_kw.md_docs.md`](./normalization.h_kw.md_docs.md)
- [`fold.h_docs.md_docs.md`](./fold.h_docs.md_docs.md)
- [`distance.h_kw.md_docs.md`](./distance.h_kw.md_docs.md)
- [`transformercoder.h_kw.md_docs.md`](./transformercoder.h_kw.md_docs.md)
- [`instancenorm.h_docs.md_docs.md`](./instancenorm.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `batchnorm.h_docs.md_docs.md`
- **Keyword Index**: `batchnorm.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
