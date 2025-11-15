# Documentation: `docs/aten/src/ATen/native/MaxPooling.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/MaxPooling.h_docs.md`
- **Size**: 5,813 bytes (5.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/MaxPooling.h`

## File Metadata

- **Path**: `aten/src/ATen/native/MaxPooling.h`
- **Size**: 3,268 bytes (3.19 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Pool.h>

namespace at::native {

inline void check_max_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {

  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 3,
      "max_pool1d() Expected 2D or 3D input tensor, but got ", self.sym_sizes());
  TORCH_CHECK(
      kernel_size.size() == 1,
      "max_pool1d() kernel_size must be an int, list of ints or tuple of ints of size 1 but got size ",
      kernel_size.size());
  TORCH_CHECK(
      stride.empty() || stride.size() == 1,
      "max_pool1d() stride must be None, an int, list of ints, or tuple of ints of size 1 but got size ",
      stride.size());
  TORCH_CHECK(
      padding.size() == 1,
      "max_pool1d() padding must be an int, list of ints, or tuple of ints of size 1 but got size ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 1,
      "max_pool1d() dilation must be an int, list of ints or tuple of ints of size 1 but got size ",
      dilation.size());

  // If stride=None then set it to kernel_size
  if (stride.empty()) {
    stride = kernel_size;
  }

  TORCH_CHECK(
      kernel_size[0] > 0,
      "max_pool1d() kernel_size must be greater than zero, but got ",
      kernel_size[0]);
  TORCH_CHECK(
      stride[0] > 0, "max_pool1d() stride must be greater than zero, but got ", stride[0]);
  TORCH_CHECK(
      padding[0] >= 0, "max_pool1d() padding must be non-negative, but got ", padding[0]);
  TORCH_CHECK(
      padding[0] <= kernel_size[0] / 2,
      "max_pool1d() padding should be at most half of kernel size, but got padding=",
      padding[0],
      " and kernel_size=",
      kernel_size[0]);
  TORCH_CHECK(
      dilation[0] > 0, "max_pool1d() dilation must be greater than zero, but got ", dilation[0]);

  const int64_t OW = pooling_output_shape(self.sym_size(-1).guard_int(__FILE__, __LINE__), kernel_size[0], padding[0], stride[0], dilation[0], ceil_mode);
  TORCH_CHECK(OW > 0, "max_pool1d() Invalid computed output size: ", OW);
}

// TODO(Heitor) Template by dimension
struct PoolingParams1D {
  int64_t NB; // Number of batches
  int64_t NC; // Number of channels
  int64_t IW; // Input width
  int64_t OW; // Output width
  int64_t KW; // Kernel width
  int64_t SJ; // Column stride
  int64_t PJ; // Column padding
  int64_t DJ; // Column dilation

  // Return index of input element for the given kernel and output index
  inline int64_t index(int64_t kj, int64_t oj) const {
    return oj * SJ + kj * DJ - PJ;
  }

  // Return index of first output within bounds for this kernel index
  inline int64_t valid_output_start(int64_t kj) const {
    int64_t ij = index(kj, 0);;
    return ij < 0 ? at::divup(-ij, SJ) : 0;
  }

  // Return index one past last output within bounds for this kernel index
  inline int64_t valid_output_end(int64_t kj) const {
    int64_t ij = index(kj, OW - 1);
    return ij >= IW ? OW - at::divup(ij - (IW - 1), SJ) : OW;
  }
};

using pooling_fn = void (*)(Tensor&, const Tensor&, const PoolingParams1D&);

DECLARE_DISPATCH(pooling_fn, max_pool1d_stub)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `PoolingParams1D`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Parallel.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/Pool.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `MaxPooling.h_docs.md`
- **Keyword Index**: `MaxPooling.h_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `MaxPooling.h_docs.md_docs.md`
- **Keyword Index**: `MaxPooling.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
