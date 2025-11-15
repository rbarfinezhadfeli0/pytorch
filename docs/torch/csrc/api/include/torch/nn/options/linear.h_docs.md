# Documentation: `torch/csrc/api/include/torch/nn/options/linear.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/options/linear.h`
- **Size**: 2,779 bytes (2.71 KB)
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

/// Options for the `Linear` module.
///
/// Example:
/// ```
/// Linear model(LinearOptions(5, 2).bias(false));
/// ```
struct TORCH_API LinearOptions {
  LinearOptions(int64_t in_features, int64_t out_features);
  /// size of each input sample
  TORCH_ARG(int64_t, in_features);

  /// size of each output sample
  TORCH_ARG(int64_t, out_features);

  /// If set to false, the layer will not learn an additive bias. Default: true
  TORCH_ARG(bool, bias) = true;
};

// ============================================================================

/// Options for the `Flatten` module.
///
/// Example:
/// ```
/// Flatten model(FlattenOptions().start_dim(2).end_dim(4));
/// ```
struct TORCH_API FlattenOptions {
  /// first dim to flatten
  TORCH_ARG(int64_t, start_dim) = 1;
  /// last dim to flatten
  TORCH_ARG(int64_t, end_dim) = -1;
};

// ============================================================================

/// Options for the `Unflatten` module.
///
/// Note: If input tensor is named, use dimname and namedshape arguments.
///
/// Example:
/// ```
/// Unflatten unnamed_model(UnflattenOptions(0, {2, 2}));
/// Unflatten named_model(UnflattenOptions("B", {{"B1", 2}, {"B2", 2}}));
/// ```
struct TORCH_API UnflattenOptions {
  typedef std::vector<std::pair<std::string, int64_t>> namedshape_t;

  UnflattenOptions(int64_t dim, std::vector<int64_t> sizes);
  UnflattenOptions(const char* dimname, namedshape_t namedshape);
  UnflattenOptions(std::string dimname, namedshape_t namedshape);

  /// dim to unflatten
  TORCH_ARG(int64_t, dim);
  /// name of dim to unflatten, for use with named tensors
  TORCH_ARG(std::string, dimname);
  /// new shape of unflattened dim
  TORCH_ARG(std::vector<int64_t>, sizes);
  /// new shape of unflattened dim with names, for use with named tensors
  TORCH_ARG(namedshape_t, namedshape);
};

// ============================================================================

/// Options for the `Bilinear` module.
///
/// Example:
/// ```
/// Bilinear model(BilinearOptions(3, 2, 4).bias(false));
/// ```
struct TORCH_API BilinearOptions {
  BilinearOptions(
      int64_t in1_features,
      int64_t in2_features,
      int64_t out_features);
  /// The number of features in input 1 (columns of the input1 matrix).
  TORCH_ARG(int64_t, in1_features);
  /// The number of features in input 2 (columns of the input2 matrix).
  TORCH_ARG(int64_t, in2_features);
  /// The number of output features to produce (columns of the output matrix).
  TORCH_ARG(int64_t, out_features);
  /// Whether to learn and add a bias after the bilinear transformation.
  TORCH_ARG(bool, bias) = true;
};

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

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
- [`transformer.h_docs.md`](./transformer.h_docs.md)


## Cross-References

- **File Documentation**: `linear.h_docs.md`
- **Keyword Index**: `linear.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
