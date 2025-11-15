# Documentation: `docs/aten/src/ATen/native/AffineGridGenerator.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/AffineGridGenerator.cpp_docs.md`
- **Size**: 7,197 bytes (7.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/AffineGridGenerator.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/AffineGridGenerator.cpp`
- **Size**: 4,554 bytes (4.45 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/affine_grid_generator_backward_native.h>
#include <ATen/ops/affine_grid_generator_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::native {

static at::Tensor linspace_from_neg_one(const Tensor& grid, int64_t num_steps,
                                 bool align_corners) {
  if (num_steps <= 1) {
    return at::tensor(0, grid.options());
  }
  auto range = at::linspace(-1, 1, num_steps, grid.options());
  if (!align_corners) {
    range = range * (num_steps - 1) / num_steps;
  }
  return range;
}

static Tensor make_base_grid_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = at::empty({N, H, W, 3}, theta.options());

  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
  base_grid.select(-1, 2).fill_(1);

  return base_grid;
}

static Tensor make_base_grid_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = at::empty({N, D, H, W, 4}, theta.options());

  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
  base_grid.select(-1, 2).copy_(linspace_from_neg_one(theta, D, align_corners).unsqueeze_(-1).unsqueeze_(-1));
  base_grid.select(-1, 3).fill_(1);

  return base_grid;
}

static Tensor affine_grid_generator_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  Tensor base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners);
  auto grid = base_grid.view({N, H * W, 3}).bmm(theta.transpose(1, 2));
  return grid.view({N, H, W, 2});
}

static Tensor affine_grid_generator_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  Tensor base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners);
  auto grid = base_grid.view({N, D * H * W, 4}).bmm(theta.transpose(1, 2));
  return grid.view({N, D, H, W, 3});
}

Tensor affine_grid_generator(const Tensor& theta, IntArrayRef size, bool align_corners) {
  TORCH_CHECK(
      size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
  if (size.size() == 4) {
    return affine_grid_generator_4D(
        theta, size[0], size[1], size[2], size[3], align_corners);
  } else {
    return affine_grid_generator_5D(
        theta, size[0], size[1], size[2], size[3], size[4], align_corners);
  }
}

static Tensor affine_grid_generator_4D_backward(
    const Tensor& grad_grid,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = make_base_grid_4D(grad_grid, N, C, H, W, align_corners);
  AT_ASSERT(grad_grid.sizes() == IntArrayRef({N, H, W, 2}));
  auto grad_theta = base_grid.view({N, H * W, 3})
                        .transpose(1, 2)
                        .bmm(grad_grid.reshape({N, H * W, 2}));
  return grad_theta.transpose(1, 2);
}

static Tensor affine_grid_generator_5D_backward(
    const Tensor& grad_grid,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = make_base_grid_5D(grad_grid, N, C, D, H, W, align_corners);
  AT_ASSERT(grad_grid.sizes() == IntArrayRef({N, D, H, W, 3}));
  auto grad_theta = base_grid.view({N, D * H * W, 4})
                        .transpose(1, 2)
                        .bmm(grad_grid.reshape({N, D * H * W, 3}));
  return grad_theta.transpose(1, 2);
}

Tensor affine_grid_generator_backward(const Tensor& grad, IntArrayRef size, bool align_corners) {
  TORCH_CHECK(
      size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
  if (size.size() == 4) {
    return affine_grid_generator_4D_backward(
        grad, size[0], size[1], size[2], size[3], align_corners);
  } else {
    return affine_grid_generator_5D_backward(
        grad, size[0], size[1], size[2], size[3], size[4], align_corners);
  }
}

}  // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/TensorOperators.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/affine_grid_generator_backward_native.h`
- `ATen/ops/affine_grid_generator_native.h`
- `ATen/ops/empty.h`
- `ATen/ops/linspace.h`
- `ATen/ops/tensor.h`


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

- **File Documentation**: `AffineGridGenerator.cpp_docs.md`
- **Keyword Index**: `AffineGridGenerator.cpp_kw.md`
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

- **File Documentation**: `AffineGridGenerator.cpp_docs.md_docs.md`
- **Keyword Index**: `AffineGridGenerator.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
