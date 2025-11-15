# Documentation: `torch/csrc/api/include/torch/nn/options/dropout.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/options/dropout.h`
- **Size**: 3,045 bytes (2.97 KB)
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

/// Options for the `Dropout` module.
///
/// Example:
/// ```
/// Dropout model(DropoutOptions().p(0.42).inplace(true));
/// ```
struct TORCH_API DropoutOptions {
  /* implicit */ DropoutOptions(double p = 0.5);

  /// The probability of an element to be zeroed. Default: 0.5
  TORCH_ARG(double, p) = 0.5;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

/// Options for the `Dropout2d` module.
///
/// Example:
/// ```
/// Dropout2d model(Dropout2dOptions().p(0.42).inplace(true));
/// ```
using Dropout2dOptions = DropoutOptions;

/// Options for the `Dropout3d` module.
///
/// Example:
/// ```
/// Dropout3d model(Dropout3dOptions().p(0.42).inplace(true));
/// ```
using Dropout3dOptions = DropoutOptions;

/// Options for the `AlphaDropout` module.
///
/// Example:
/// ```
/// AlphaDropout model(AlphaDropoutOptions(0.2).inplace(true));
/// ```
using AlphaDropoutOptions = DropoutOptions;

/// Options for the `FeatureAlphaDropout` module.
///
/// Example:
/// ```
/// FeatureAlphaDropout model(FeatureAlphaDropoutOptions(0.2).inplace(true));
/// ```
using FeatureAlphaDropoutOptions = DropoutOptions;

namespace functional {

/// Options for `torch::nn::functional::dropout`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout(input, F::DropoutFuncOptions().p(0.5));
/// ```
struct TORCH_API DropoutFuncOptions {
  /// The probability of an element to be zeroed. Default: 0.5
  TORCH_ARG(double, p) = 0.5;

  TORCH_ARG(bool, training) = true;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

/// Options for `torch::nn::functional::dropout2d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout2d(input, F::Dropout2dFuncOptions().p(0.5));
/// ```
using Dropout2dFuncOptions = DropoutFuncOptions;

/// Options for `torch::nn::functional::dropout3d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout3d(input, F::Dropout3dFuncOptions().p(0.5));
/// ```
using Dropout3dFuncOptions = DropoutFuncOptions;

/// Options for `torch::nn::functional::alpha_dropout`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::alpha_dropout(input,
/// F::AlphaDropoutFuncOptions().p(0.5).training(false));
/// ```
struct TORCH_API AlphaDropoutFuncOptions {
  TORCH_ARG(double, p) = 0.5;

  TORCH_ARG(bool, training) = false;

  TORCH_ARG(bool, inplace) = false;
};

/// Options for `torch::nn::functional::feature_alpha_dropout`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::feature_alpha_dropout(input,
/// F::FeatureAlphaDropoutFuncOptions().p(0.5).training(false));
/// ```
struct TORCH_API FeatureAlphaDropoutFuncOptions {
  TORCH_ARG(double, p) = 0.5;

  TORCH_ARG(bool, training) = false;

  TORCH_ARG(bool, inplace) = false;
};

} // namespace functional

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `functional`, `F`

**Classes/Structs**: `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`


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
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`transformer.h_docs.md`](./transformer.h_docs.md)


## Cross-References

- **File Documentation**: `dropout.h_docs.md`
- **Keyword Index**: `dropout.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
