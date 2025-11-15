# Documentation: `docs/aten/src/ATen/native/GridSamplerUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/GridSamplerUtils.h_docs.md`
- **Size**: 6,247 bytes (6.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/GridSamplerUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/GridSamplerUtils.h`
- **Size**: 3,705 bytes (3.62 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// See NOTE: [Tensor vs. TensorBase]
// https://github.com/pytorch/pytorch/pull/66979
#include <ATen/core/TensorBase.h>
#include <ATen/native/TensorProperties.h>
#include <ATen/native/CanUse32BitIndexMath.h>

namespace at::native {

namespace detail {

enum class GridSamplerInterpolation {Bilinear, Nearest, Bicubic};
enum class GridSamplerPadding {Zeros, Border, Reflection};

} // namespace detail

using detail::GridSamplerInterpolation;
using detail::GridSamplerPadding;

// See NOTE [ grid_sampler Native Functions ].
inline void check_grid_sampler_common(
  const TensorBase& input,
  const TensorBase& grid
) {
  auto input_opt = input.options();
  auto grid_opt = grid.options();

  TORCH_CHECK(
    input.defined(),
    "grid_sampler(): expected input to not be undefined");
  TORCH_CHECK(
    grid.defined(),
    "grid_sampler(): expected grid to not be undefined");
  TORCH_CHECK(
    input_opt.device() == grid_opt.device(),
    "grid_sampler(): expected input and grid to be on same device, but input "
    "is on ", input_opt.device(), " and grid is on ", grid_opt.device());
  TORCH_CHECK(
    input_opt.layout() == kStrided && grid_opt.layout() == kStrided,
    "grid_sampler(): expected input and grid to have torch.strided layout, but "
    "input has ", input_opt.layout(), " and grid has ", grid_opt.layout());
  TORCH_CHECK(
    input.size(0) == grid.size(0),
    "grid_sampler(): expected grid and input to have same batch size, but got "
    "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());

  for (const auto i : c10::irange(2, input.dim())) {
    TORCH_CHECK(input.size(i) > 0,
      "grid_sampler(): expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }
}

// See NOTE [ grid_sampler Native Functions ].
inline void check_grid_sampler_2d(
  const TensorBase& input,
  const TensorBase& grid
) {
  TORCH_CHECK(
    input.dim() == 4 && input.dim() == grid.dim(),
    "grid_sampler(): expected 4D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
}

// See NOTE [ grid_sampler Native Functions ].
inline void check_grid_sampler_3d(
  const TensorBase& input,
  const TensorBase& grid,
  int64_t interpolation_mode
) {
  TORCH_CHECK(
    input.dim() == 5 && input.dim() == grid.dim(),
    "grid_sampler(): expected 5D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    !(input.dim() == 5 &&
      static_cast<GridSamplerInterpolation>(interpolation_mode) ==
        GridSamplerInterpolation::Bicubic),
    "grid_sampler(): bicubic interpolation only supports 4D input");
}

// See NOTE [ grid_sampler Native Functions ].
// cudnn does not support inputs larger than 1024.
inline bool cond_cudnn_grid_sampler(
  const TensorBase& input,
  const TensorBase& grid
) {
  auto st = input.scalar_type();
  if (!(st == kDouble || st == kFloat || st == kHalf))
    return false;
  st = grid.scalar_type();
  if (!(st == kDouble || st == kFloat || st == kHalf))
    return false;
  return (
    at::native::cudnn_is_acceptable(input) &&
    at::native::cudnn_is_acceptable(grid) &&
    at::native::canUse32BitIndexMath(input) &&
    at::native::canUse32BitIndexMath(grid) &&
    input.dim() == 4 &&
    input.sym_size(1) <= 1024);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `at`

**Classes/Structs**: `GridSamplerInterpolation`, `GridSamplerPadding`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/TensorBase.h`
- `ATen/native/TensorProperties.h`
- `ATen/native/CanUse32BitIndexMath.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `GridSamplerUtils.h_docs.md`
- **Keyword Index**: `GridSamplerUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `GridSamplerUtils.h_docs.md_docs.md`
- **Keyword Index**: `GridSamplerUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
