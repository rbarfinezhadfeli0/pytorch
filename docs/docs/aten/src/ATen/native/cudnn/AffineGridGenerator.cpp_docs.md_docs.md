# Documentation: `docs/aten/src/ATen/native/cudnn/AffineGridGenerator.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cudnn/AffineGridGenerator.cpp_docs.md`
- **Size**: 6,022 bytes (5.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cudnn/AffineGridGenerator.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cudnn/AffineGridGenerator.cpp`
- **Size**: 3,160 bytes (3.09 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cudnn_affine_grid_generator_backward_native.h>
#include <ATen/ops/cudnn_affine_grid_generator_native.h>
#include <ATen/ops/empty.h>
#endif

#if !AT_CUDNN_ENABLED()

namespace at {
namespace native {

// See Note [ATen preprocessor philosophy]

Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
  TORCH_CHECK(
      false,
      "cudnn_affine_grid_generator_forward: ATen not compiled with cuDNN support");
}

Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
  TORCH_CHECK(
      false,
      "cudnn_affine_grid_generator_backward: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED()

#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cudnn/cudnn-wrapper.h>

#include <ATen/TensorUtils.h>

namespace at {
namespace native {

namespace {

void setSamplerDescriptor(
    SpatialTransformerDescriptor& desc,
    cudnnDataType_t dataType,
    int N,
    int C,
    int H,
    int W) {
  int inputSize[4] = {N, C, H, W};
  desc.set(dataType, 4, inputSize);
}

} // namespace

Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta_t,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
  auto theta_t_contig = theta_t.contiguous();
  TensorArg theta{theta_t_contig, "theta", 1};
  CheckedFrom c = "cudnn_affine_grid_generator_forward";
  checkContiguous(c, theta);
  checkSize(c, theta, {N, 2, 3});

  auto grid_t = at::empty({0}, theta->options());
  grid_t.resize_({N, H, W, 2});

  auto dataType = getCudnnDataType(*theta);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorForward(
      getCudnnHandle(), desc.desc(), theta->data_ptr(), grid_t.data_ptr()));
  return grid_t;
}

Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_grid_t,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
  auto grad_grid_contig = grad_grid_t.contiguous();
  TensorArg grad_grid{grad_grid_contig, "grad_grid", 1};
  CheckedFrom c = "cudnn_affine_grid_generator_backward";
  checkContiguous(c, grad_grid);
  checkSize(c, grad_grid, {N, H, W, 2});

  auto grad_theta_t = at::empty({0}, grad_grid->options());
  grad_theta_t.resize_({N, 2, 3});

  auto dataType = getCudnnDataType(grad_theta_t);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorBackward(
      getCudnnHandle(),
      desc.desc(),
      grad_grid->data_ptr(),
      grad_theta_t.data_ptr()));
  return grad_theta_t;
}

} // namespace native
} // namespace at

#endif // AT_CUDNN_ENABLED()

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `native`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/core/Tensor.h`
- `ATen/cuda/CUDAConfig.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/cudnn_affine_grid_generator_backward_native.h`
- `ATen/ops/cudnn_affine_grid_generator_native.h`
- `ATen/ops/empty.h`
- `ATen/cuda/Exceptions.h`
- `ATen/cudnn/Descriptors.h`
- `ATen/cudnn/Handle.h`
- `ATen/cudnn/Types.h`
- `ATen/cudnn/Utils.h`
- `ATen/cudnn/cudnn-wrapper.h`
- `ATen/TensorUtils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`aten/src/ATen/native/cudnn`):

- [`MHA.cpp_docs.md`](./MHA.cpp_docs.md)
- [`Conv_v7.cpp_docs.md`](./Conv_v7.cpp_docs.md)
- [`RNN.cpp_docs.md`](./RNN.cpp_docs.md)
- [`ConvShared.cpp_docs.md`](./ConvShared.cpp_docs.md)
- [`MHA.h_docs.md`](./MHA.h_docs.md)
- [`ConvShared.h_docs.md`](./ConvShared.h_docs.md)
- [`RNNUtils.h_docs.md`](./RNNUtils.h_docs.md)
- [`BatchNorm.h_docs.md`](./BatchNorm.h_docs.md)
- [`Conv_v8.cpp_docs.md`](./Conv_v8.cpp_docs.md)
- [`GridSampler.cpp_docs.md`](./GridSampler.cpp_docs.md)


## Cross-References

- **File Documentation**: `AffineGridGenerator.cpp_docs.md`
- **Keyword Index**: `AffineGridGenerator.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cudnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/aten/src/ATen/native/cudnn`):

- [`BatchNorm.cpp_kw.md_docs.md`](./BatchNorm.cpp_kw.md_docs.md)
- [`GridSampler.cpp_kw.md_docs.md`](./GridSampler.cpp_kw.md_docs.md)
- [`ConvShared.cpp_docs.md_docs.md`](./ConvShared.cpp_docs.md_docs.md)
- [`RNN.cpp_docs.md_docs.md`](./RNN.cpp_docs.md_docs.md)
- [`MHA.cpp_kw.md_docs.md`](./MHA.cpp_kw.md_docs.md)
- [`Conv_v8.cpp_kw.md_docs.md`](./Conv_v8.cpp_kw.md_docs.md)
- [`Conv_v7.cpp_kw.md_docs.md`](./Conv_v7.cpp_kw.md_docs.md)
- [`AffineGridGenerator.cpp_kw.md_docs.md`](./AffineGridGenerator.cpp_kw.md_docs.md)
- [`BatchNorm.h_kw.md_docs.md`](./BatchNorm.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `AffineGridGenerator.cpp_docs.md_docs.md`
- **Keyword Index**: `AffineGridGenerator.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
