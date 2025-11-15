# Documentation: `docs/torch/csrc/api/include/torch/nn/options/padding.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/options/padding.h_docs.md`
- **Size**: 9,253 bytes (9.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/options/padding.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/options/padding.h`
- **Size**: 6,835 bytes (6.67 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch::nn {

/// Options for a `D`-dimensional ReflectionPad module.
template <size_t D>
struct TORCH_API ReflectionPadOptions {
  ReflectionPadOptions(ExpandingArray<D * 2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// If it is `int`, uses the same padding in all boundaries.
  /// If it is a 2-`tuple` (for ReflectionPad1d), uses (padding_left,
  /// padding_right). If it is a 4-`tuple` (for ReflectionPad2d), uses
  /// (padding_left, padding_right, padding_top, padding_bottom). If it is a
  /// 6-`tuple` (for ReflectionPad3d), uses (padding_left, padding_right,
  /// padding_top, padding_bottom, padding_front, padding_back).

  TORCH_ARG(ExpandingArray<D * 2>, padding);
};

/// `ReflectionPadOptions` specialized for the `ReflectionPad1d` module.
///
/// Example:
/// ```
/// ReflectionPad1d model(ReflectionPad1dOptions({3, 1}));
/// ```
using ReflectionPad1dOptions = ReflectionPadOptions<1>;

/// `ReflectionPadOptions` specialized for the `ReflectionPad2d` module.
///
/// Example:
/// ```
/// ReflectionPad2d model(ReflectionPad2dOptions({1, 1, 2, 0}));
/// ```
using ReflectionPad2dOptions = ReflectionPadOptions<2>;

/// `ReflectionPadOptions` specialized for the `ReflectionPad3d` module.
///
/// Example:
/// ```
/// ReflectionPad3d model(ReflectionPad3dOptions({1, 1, 2, 0, 1, 1}));
/// ```
using ReflectionPad3dOptions = ReflectionPadOptions<3>;

// ============================================================================

/// Options for a `D`-dimensional ReplicationPad module.
template <size_t D>
struct TORCH_API ReplicationPadOptions {
  ReplicationPadOptions(ExpandingArray<D * 2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// - If it is `int`, uses the same padding in all boundaries.
  /// - If it is a 2-`tuple` (for ReplicationPad1d), uses (padding_left,
  /// padding_right).
  /// - If it is a 4-`tuple` (for ReplicationPad2d), uses (padding_left,
  /// padding_right, padding_top, padding_bottom).
  /// - If it is a 6-`tuple` (for ReplicationPad3d), uses
  ///   (padding_left, padding_right, padding_top, padding_bottom,
  ///   padding_front, padding_back).
  TORCH_ARG(ExpandingArray<D * 2>, padding);
};

/// `ReplicationPadOptions` specialized for the `ReplicationPad1d` module.
///
/// Example:
/// ```
/// ReplicationPad1d model(ReplicationPad1dOptions({3, 1}));
/// ```
using ReplicationPad1dOptions = ReplicationPadOptions<1>;

/// `ReplicationPadOptions` specialized for the `ReplicationPad2d` module.
///
/// Example:
/// ```
/// ReplicationPad2d model(ReplicationPad2dOptions({1, 1, 2, 0}));
/// ```
using ReplicationPad2dOptions = ReplicationPadOptions<2>;

/// `ReplicationPadOptions` specialized for the `ReplicationPad3d` module.
///
/// Example:
/// ```
/// ReplicationPad3d model(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}));
/// ```
using ReplicationPad3dOptions = ReplicationPadOptions<3>;

// ============================================================================

template <size_t D>
struct TORCH_API ZeroPadOptions {
  ZeroPadOptions(ExpandingArray<D * 2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// - If it is `int`, uses the same padding in all boundaries.
  /// - If it is a 2-`tuple` (for ZeroPad1d), uses (padding_left,
  /// padding_right).
  /// - If it is a 4-`tuple` (for ZeroPad2d), uses (padding_left, padding_right,
  /// padding_top, padding_bottom).
  /// - If it is a 6-`tuple` (for ZeroPad3d), uses
  ///   (padding_left, padding_right, padding_top, padding_bottom,
  ///   padding_front, padding_back).
  TORCH_ARG(ExpandingArray<D * 2>, padding);
};

/// `ZeroPadOptions` specialized for the `ZeroPad1d` module.
///
/// Example:
/// ```
/// ConstantPad1d model(ConstantPad1dOptions({3, 1});
/// ```
using ZeroPad1dOptions = ZeroPadOptions<1>;

/// `ZeroPadOptions` specialized for the `ZeroPad2d` module.
///
/// Example:
/// ```
/// ConstantPad2d model(ConstantPad2dOptions({1, 1, 2, 0});
/// ```
using ZeroPad2dOptions = ZeroPadOptions<2>;

/// `ZeroPadOptions` specialized for the `ZeroPad3d` module.
///
/// Example:
/// ```
/// ConstantPad3d model(ConstantPad3dOptions({1, 2, 1, 2, 1, 2});
/// ```
using ZeroPad3dOptions = ZeroPadOptions<3>;

// ============================================================================

/// Options for a `D`-dimensional ConstantPad module.
template <size_t D>
struct TORCH_API ConstantPadOptions {
  ConstantPadOptions(ExpandingArray<D * 2> padding, double value)
      : padding_(padding), value_(value) {}

  /// The size of the padding.
  /// - If it is `int`, uses the same padding in all boundaries.
  /// - If it is a 2-`tuple` (for ConstantPad1d), uses (padding_left,
  /// padding_right).
  /// - If it is a 4-`tuple` (for ConstantPad2d), uses (padding_left,
  /// padding_right, padding_top, padding_bottom).
  /// - If it is a 6-`tuple` (for ConstantPad3d), uses
  ///   (padding_left, padding_right, padding_top, padding_bottom,
  ///   padding_front, padding_back).
  TORCH_ARG(ExpandingArray<D * 2>, padding);

  /// Fill value for constant padding.
  TORCH_ARG(double, value);
};

/// `ConstantPadOptions` specialized for the `ConstantPad1d` module.
///
/// Example:
/// ```
/// ConstantPad1d model(ConstantPad1dOptions({3, 1}, 3.5));
/// ```
using ConstantPad1dOptions = ConstantPadOptions<1>;

/// `ConstantPadOptions` specialized for the `ConstantPad2d` module.
///
/// Example:
/// ```
/// ConstantPad2d model(ConstantPad2dOptions({3, 0, 2, 1}, 3.5));
/// ```
using ConstantPad2dOptions = ConstantPadOptions<2>;

/// `ConstantPadOptions` specialized for the `ConstantPad3d` module.
///
/// Example:
/// ```
/// ConstantPad3d model(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5));
/// ```
using ConstantPad3dOptions = ConstantPadOptions<3>;

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::pad`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::pad(input, F::PadFuncOptions({1, 2, 2, 1, 1,
/// 2}).mode(torch::kReplicate));
/// ```
struct TORCH_API PadFuncOptions {
  typedef std::variant<
      enumtype::kConstant,
      enumtype::kReflect,
      enumtype::kReplicate,
      enumtype::kCircular>
      mode_t;

  PadFuncOptions(std::vector<int64_t> pad);

  /// m-elements tuple, where m/2 <= input dimensions and m is even.
  TORCH_ARG(std::vector<int64_t>, pad);

  /// "constant", "reflect", "replicate" or "circular". Default: "constant"
  TORCH_ARG(mode_t, mode) = torch::kConstant;

  /// fill value for "constant" padding. Default: 0
  TORCH_ARG(double, value) = 0;
};

} // namespace functional

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `functional`, `F`

**Classes/Structs**: `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/options`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/arg.h`
- `torch/csrc/Export.h`
- `torch/enum.h`
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
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`transformer.h_docs.md`](./transformer.h_docs.md)


## Cross-References

- **File Documentation**: `padding.h_docs.md`
- **Keyword Index**: `padding.h_kw.md`
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

- **File Documentation**: `padding.h_docs.md_docs.md`
- **Keyword Index**: `padding.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
