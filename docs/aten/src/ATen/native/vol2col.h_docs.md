# Documentation: `aten/src/ATen/native/vol2col.h`

## File Metadata

- **Path**: `aten/src/ATen/native/vol2col.h`
- **Size**: 3,555 bytes (3.47 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <cstring>

namespace at::native {

template <typename T>
void vol2col(
    const T* data_vol,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t depth_col,
    const int64_t height_col,
    const int64_t width_col,
    const int64_t kT,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pT,
    const int64_t pH,
    const int64_t pW,
    const int64_t dT,
    const int64_t dH,
    const int64_t dW,
    const int64_t dilationT,
    const int64_t dilationH,
    const int64_t dilationW,
    T* data_col) {
  int64_t c, t, h, w;
  int64_t channels_col = channels * kT * kernel_height * kernel_width;
  for (c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % kernel_width;
    int64_t h_offset = (c / kernel_width) % kernel_height;
    int64_t t_offset = (c / kernel_width / kernel_height) % kT;
    int64_t c_vol = c / kT / kernel_height / kernel_width;
    for (t = 0; t < depth_col; ++t) {
      int64_t t_pad = t * dT - pT + t_offset * dilationT;
      for (h = 0; h < height_col; ++h) {
        int64_t h_pad = h * dH - pH + h_offset * dilationH;
        for (w = 0; w < width_col; ++w) {
          int64_t w_pad = w * dW - pW + w_offset * dilationW;
          if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                data_vol
                    [((c_vol * depth + t_pad) * height + h_pad) * width +
                     w_pad];
          else
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                0;
        }
      }
    }
  }
}

template <typename T>
void col2vol(
    const T* data_col,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t out_depth,
    const int64_t out_height,
    const int64_t out_width,
    const int64_t kT,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pT,
    const int64_t pH,
    const int64_t pW,
    const int64_t dT,
    const int64_t dH,
    const int64_t dW,
    const int64_t dilationT,
    const int64_t dilationH,
    const int64_t dilationW,
    T* data_vol) {
  memset(data_vol, 0, sizeof(T) * depth * height * width * channels);
  int64_t depth_col = out_depth;
  int64_t height_col = out_height;
  int64_t width_col = out_width;
  int64_t channels_col = channels * kT * kernel_height * kernel_width;
  for (int64_t c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % kernel_width;
    int64_t h_offset = (c / kernel_width) % kernel_height;
    int64_t t_offset = (c / kernel_width / kernel_height) % kT;
    int64_t c_vol = c / kT / kernel_height / kernel_width;
    for (int64_t t = 0; t < depth_col; ++t) {
      int64_t t_pad = t * dT - pT + t_offset * dilationT;
      for (int64_t h = 0; h < height_col; ++h) {
        int64_t h_pad = h * dH - pH + h_offset * dilationH;
        for (int64_t w = 0; w < width_col; ++w) {
          int64_t w_pad = w * dW - pW + w_offset * dilationW;
          if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_vol
                [((c_vol * depth + t_pad) * height + h_pad) * width + w_pad] +=
                data_col
                    [((c * depth_col + t) * height_col + h) * width_col + w];
        }
      }
    }
  }
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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

- `cstring`


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

- **File Documentation**: `vol2col.h_docs.md`
- **Keyword Index**: `vol2col.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
