# Documentation: `aten/src/ATen/native/im2col.h`

## File Metadata

- **Path**: `aten/src/ATen/native/im2col.h`
- **Size**: 5,237 bytes (5.11 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#include <algorithm>

namespace at::native {

template <typename T>
static void im2col(
    const T* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_col,
    bool is_channels_last = false) {
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;

  if (is_channels_last) {
    at::parallel_for(0, height_col * width_col, 0, [&](int64_t begin, int64_t end) {
      int64_t h_col{0}, w_col{0};
      data_index_init(begin, h_col, height_col, w_col, width_col);

      for (const auto i_col : c10::irange(begin, end)) {
        for (const auto h_offset : c10::irange(kernel_h)) {
          int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
          for (const auto w_offset : c10::irange(kernel_w)) {
            int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

            const T* slice_im = data_im + (h_im * width + w_im) * channels;
            T* slice_col = data_col + (i_col * kernel_h * kernel_w + h_offset * kernel_w + w_offset) * channels;

            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
              std::copy_n(slice_im, channels, slice_col);
            } else {
              std::fill_n(slice_col, channels, T(0));
            }
          }
        }

        // move the next index
        data_index_step(h_col, height_col, w_col, width_col);
      }
    });
  } else {
    at::parallel_for(0, channels_col, 0, [&](int64_t begin, int64_t end) {
      int64_t c_im{0}, h_offset{0}, w_offset{0};
      data_index_init(begin, c_im, channels, h_offset, kernel_h, w_offset, kernel_w);

      for (const auto c_col : c10::irange(begin, end)) {
        for (const auto h_col : c10::irange(height_col)) {
          int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
          for (const auto w_col : c10::irange(width_col)) {
            int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
            data_col[(c_col * height_col + h_col) * width_col + w_col] =
                (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? c10::load(&(data_im[(c_im * height + h_im) * width + w_im]))
                : static_cast<T>(0);
          }
        }

        // move to the next index
        data_index_step(c_im, channels, h_offset, kernel_h, w_offset, kernel_w);
      }
    });
  }
}

template <typename T>
static void col2im(
    const T* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_im,
    bool is_channels_last = false) {
  std::fill_n(data_im, height * width * channels, T(0));

  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;

  if (is_channels_last) {
    for (const auto h_col : c10::irange(height_col)) {
      for (const auto w_col : c10::irange(width_col)) {
        for (const auto h_offset : c10::irange(kernel_h)) {
          int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
          for (const auto w_offset : c10::irange(kernel_w)) {
            int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

            T* slice_im = data_im + (h_im * width + w_im) * channels;
            const T* slice_col = data_col + ((h_col * width_col + w_col) * kernel_h * kernel_w
                + h_offset * kernel_w + w_offset) * channels;

            if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
              std::transform(slice_col, slice_col + channels, slice_im, slice_im, std::plus<T>());
            }
          }
        }
      }
    }
  } else {
    for (const auto c_col : c10::irange(channels_col)) {
      int64_t w_offset = c_col % kernel_w;
      int64_t h_offset = (c_col / kernel_w) % kernel_h;
      int64_t c_im = c_col / kernel_h / kernel_w;

      for (const auto h_col : c10::irange(height_col)) {
        int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        for (const auto w_col : c10::irange(width_col)) {
          int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

          if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
            data_im[(c_im * height + h_im) * width + w_im] +=
                data_col[(c_col * height_col + h_col) * width_col + w_col];
        }
      }
    }
  }
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

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
- `ATen/TensorUtils.h`
- `ATen/Utils.h`
- `ATen/Parallel.h`
- `ATen/native/cpu/utils.h`
- `c10/util/irange.h`
- `algorithm`


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

- **File Documentation**: `im2col.h_docs.md`
- **Keyword Index**: `im2col.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
