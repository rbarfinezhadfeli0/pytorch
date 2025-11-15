# Documentation: `docs/aten/src/ATen/native/cpu/Unfold2d.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/Unfold2d.cpp_docs.md`
- **Size**: 17,738 bytes (17.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/Unfold2d.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/Unfold2d.cpp`
- **Size**: 15,026 bytes (14.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Unfold2d.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>
#include <ATen/native/cpu/utils.h>
#include <cmath>

namespace at::native {

namespace {

template <typename scalar_t>
inline void cadd(
    scalar_t* z,
    const scalar_t* x,
    const scalar_t* y,
    int64_t n) {
  using Vec = vec::Vectorized<scalar_t>;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  char* ptrs[] = {reinterpret_cast<char*>(z),
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                  reinterpret_cast<char*>(const_cast<scalar_t*>(x)),
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                  reinterpret_cast<char*>(const_cast<scalar_t*>(y))};
  vectorized_loop(
      ptrs,
      n,
      -1,
      [](scalar_t x, scalar_t y) -> scalar_t { return x + y; },
      [](Vec x, Vec y) -> Vec { return x + y; });
}

template <typename scalar_t>
void unfolded2d_acc(
    scalar_t* finput_data,
    scalar_t* input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  at::parallel_for(0, n_input_plane, 0, [&](int64_t start, int64_t end) {
    for (const auto nip : c10::irange(start, end)) {
      for (int64_t kh = 0; kh < kH; kh++) {
        for (int64_t kw = 0; kw < kW; kw++) {
          scalar_t* src = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          scalar_t* dst =
              input_data + nip * ((size_t)input_height * input_width);
          if (padW > 0 || padH > 0) {
            for (int64_t y = 0; y < output_height; y++) {
              auto iy = y * dH - padH + kh;
              if (iy < 0 || iy >= input_height) {
              } else {
                if (dW == 1) {
                  auto ix = 0 - padW + kw;
                  auto lpad = std::max<int64_t>(0, padW - kw);
                  auto rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + lpad;
                  cadd(
                      dst_slice,
                      dst_slice,
                      src + (size_t)y * output_width + lpad,
                      output_width - lpad - rpad);
                } else {
                  for (int64_t x = 0; x < output_width; x++) {
                    auto ix = x * dW - padW + kw;
                    if (ix < 0 || ix >= input_width) {
                    } else {
                      scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                      *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                    }
                  }
                }
              }
            }
          } else {
            for (int64_t y = 0; y < output_height; y++) {
              auto iy = y * dH + kh;
              auto ix = 0 + kw;
              if (dW == 1) {
                scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                cadd(
                    dst_slice,
                    dst_slice,
                    src + (size_t)y * output_width,
                    output_width);
              } else {
                for (int64_t x = 0; x < output_width; x++) {
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + x * dW;
                  *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                }
              }
            }
          }
        }
      }
    }
  });
}

template <typename scalar_t>
void unfolded2d_acc_channels_last(
    scalar_t* finput_data,
    scalar_t* input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {

  for (int64_t y = 0; y < output_height; y++) {
    for (int64_t x = 0; x < output_width; x++) {
      scalar_t* src = finput_data + y * output_width * kH * kW * n_input_plane + x * kH * kW * n_input_plane;
      scalar_t* dst = input_data;

      if (padW > 0 || padH > 0) {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH - padH + kh;
            int64_t ix = x * dW - padW + kw;
            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) {
            } else {
              scalar_t* dst_slice = dst + iy * input_width * n_input_plane + ix * n_input_plane;
              scalar_t* src_slice = src + kh * kW * n_input_plane + kw * n_input_plane;
              cadd(dst_slice,
                   dst_slice,
                   src_slice,
                   n_input_plane);
            }
          }
        }
      } else {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH + kh;
            int64_t ix = x * dW + kw;
            scalar_t* dst_slice = dst + iy * input_width * n_input_plane + ix * n_input_plane;
            scalar_t* src_slice = src + kh * kW * n_input_plane + kw * n_input_plane;
            cadd(dst_slice,
                 dst_slice,
                 src_slice,
                 n_input_plane);
          }
        }
      }
    }
  }
}

/* note: due to write issues, this one cannot be parallelized as well as
 * unfolded2d_copy */
#if defined(__GNUC__) && __GNUC__ == 14 && defined(__ARM_FEATURE_SVE) && !defined(__ARM_FEATURE_BF16)
// Workaround for gcc-14.2.0 ICE during RTL pass: vregs when compiling for SVE without BF16
__attribute__((optimize("no-tree-vectorize")))
#endif
void unfolded2d_acc_kernel(
    ScalarType dtype,
    void *finput_data,
    void *input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last) {
  // This function assumes that
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t

  if (is_channels_last) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_acc_channels_last", [&] {
      unfolded2d_acc_channels_last(
          static_cast<scalar_t*>(finput_data),
          static_cast<scalar_t*>(input_data),
          kH, kW,
          dH, dW,
          padH, padW,
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width);
     });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_acc", [&] {
      unfolded2d_acc(
          static_cast<scalar_t*>(finput_data),
          static_cast<scalar_t*>(input_data),
          kH, kW,
          dH, dW,
          padH, padW,
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width);
      });
  }
}

template <typename scalar_t>
void unfolded2d_copy(
    const scalar_t* input_data,
    scalar_t* finput_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  at::parallel_for(
      0, n_input_plane * kH * kW, 0, [&](int64_t start, int64_t end) {
        for (const auto k : c10::irange(start, end)) {
          int64_t nip = k / (kH * kW);
          int64_t rest = k % (kH * kW);
          int64_t kh = rest / kW;
          int64_t kw = rest % kW;
          scalar_t* dst = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          const scalar_t* src =
              input_data + nip * ((size_t)input_height * input_width);
          if (padW > 0 || padH > 0) {
            for (int64_t y = 0; y < output_height; y++) {
              auto iy = y * dH - padH + kh;
              if (iy < 0 || iy >= input_height) {
                memset(
                    dst + (size_t)y * output_width,
                    0,
                    sizeof(scalar_t) * output_width);
              } else {
                if (dW == 1) {
                  auto ix = 0 - padW + kw;
                  auto lpad = std::max<int64_t>(0, padW - kw);
                  auto rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                  if (output_width - rpad - lpad <= 0) {
                    memset(
                        dst + (size_t)y * output_width,
                        0,
                        sizeof(scalar_t) * output_width);
                  } else {
                    if (lpad > 0)
                      memset(
                          dst + (size_t)y * output_width,
                          0,
                          sizeof(scalar_t) * lpad);
                    memcpy(
                        dst + (size_t)y * output_width + lpad,
                        src + (size_t)iy * input_width + ix + lpad,
                        sizeof(scalar_t) * (output_width - rpad - lpad));
                    if (rpad > 0)
                      memset(
                          dst + (size_t)y * output_width + output_width - rpad,
                          0,
                          sizeof(scalar_t) * rpad);
                  }
                } else {
                  for (int64_t x = 0; x < output_width; x++) {
                    auto ix = x * dW - padW + kw;
                    if (ix < 0 || ix >= input_width)
                      memset(
                          dst + (size_t)y * output_width + x,
                          0,
                          sizeof(scalar_t) * 1);
                    else
                      memcpy(
                          dst + (size_t)y * output_width + x,
                          src + (size_t)iy * input_width + ix,
                          sizeof(scalar_t) * 1);
                  }
                }
              }
            }
          } else {
            for (int64_t y = 0; y < output_height; y++) {
              auto iy = y * dH + kh;
              auto ix = 0 + kw;
              if (dW == 1)
                memcpy(
                    dst + (size_t)y * output_width,
                    src + (size_t)iy * input_width + ix,
                    sizeof(scalar_t) * output_width);
              else {
                for (int64_t x = 0; x < output_width; x++)
                  memcpy(
                      dst + (size_t)y * output_width + x,
                      src + (size_t)iy * input_width + ix + x * dW,
                      sizeof(scalar_t) * 1);
              }
            }
          }
        }
      });
}

template <typename scalar_t>
void unfolded2d_copy_channels_last(
    const scalar_t* input_data,
    scalar_t* finput_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  at::parallel_for(0, output_height * output_width, 0, [&](int64_t start, int64_t end) {
    int64_t y = 0;
    int64_t x = 0;
    data_index_init(start, y, output_height, x, output_width);

    for (const auto k [[maybe_unused]] : c10::irange(start, end)) {
      scalar_t* dst = finput_data + y * output_width * kH * kW * n_input_plane +
          x * kH * kW * n_input_plane;
      const scalar_t* src = input_data;

      if (padW > 0 || padH > 0) {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH - padH + kh;
            int64_t ix = x * dW - padW + kw;
            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) {
              memset(dst + kh * kW * n_input_plane + kw * n_input_plane,
                    0,
                    sizeof(scalar_t) * n_input_plane);
            } else {
              memcpy(dst + kh * kW * n_input_plane + kw * n_input_plane,
                     src + iy * input_width * n_input_plane + ix * n_input_plane,
                     sizeof(scalar_t) * n_input_plane);
            }
          }
        }
      } else {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH + kh;
            int64_t ix = x * dW + kw;
            memcpy(dst + kh * kW * n_input_plane + kw * n_input_plane,
                   src + iy * input_width * n_input_plane + ix * n_input_plane,
                   sizeof(scalar_t) * n_input_plane);
          }
        }
      }
      // move on to next output index
      data_index_step(y, output_height, x, output_width);
    }
  });
}

void unfolded2d_copy_kernel(
    ScalarType dtype,
    void *finput_data,
    const void *input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last) {
  // This function assumes that
  // kH*kW does not overflow an int
  // n_input_plane*kH*kW does not overflow a int64_t
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t

  if (is_channels_last) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_copy_channels_last", [&] {
      unfolded2d_copy_channels_last(
          static_cast<const scalar_t*>(input_data),
          static_cast<scalar_t*>(finput_data),
            kH, kW,
            dH, dW,
            padH, padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_copy", [&] {
      unfolded2d_copy(
          static_cast<const scalar_t*>(input_data),
          static_cast<scalar_t*>(finput_data),
            kH, kW,
            dH, dW,
            padH, padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
    });
  }
}

} // namespace

REGISTER_DISPATCH(unfolded2d_copy_stub, &unfolded2d_copy_kernel)
REGISTER_DISPATCH(unfolded2d_acc_stub, &unfolded2d_acc_kernel)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `REGISTER_DISPATCH`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Dispatch.h`
- `ATen/Parallel.h`
- `ATen/cpu/vec/vec.h`
- `ATen/native/Unfold2d.h`
- `ATen/native/cpu/Loops.h`
- `c10/util/irange.h`
- `ATen/native/cpu/utils.h`
- `cmath`


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

Files in the same folder (`aten/src/ATen/native/cpu`):

- [`UpSampleKernelAVXAntialias.h_docs.md`](./UpSampleKernelAVXAntialias.h_docs.md)
- [`SparseFactories.cpp_docs.md`](./SparseFactories.cpp_docs.md)
- [`UnfoldBackwardKernel.cpp_docs.md`](./UnfoldBackwardKernel.cpp_docs.md)
- [`int8mm_kernel.cpp_docs.md`](./int8mm_kernel.cpp_docs.md)
- [`LerpKernel.cpp_docs.md`](./LerpKernel.cpp_docs.md)
- [`UpSampleKernel.cpp_docs.md`](./UpSampleKernel.cpp_docs.md)
- [`scaled_modified_bessel_k0.cpp_docs.md`](./scaled_modified_bessel_k0.cpp_docs.md)
- [`DistributionKernels.cpp_docs.md`](./DistributionKernels.cpp_docs.md)
- [`CopyKernel.cpp_docs.md`](./CopyKernel.cpp_docs.md)
- [`SampledAddmmKernel.cpp_docs.md`](./SampledAddmmKernel.cpp_docs.md)


## Cross-References

- **File Documentation**: `Unfold2d.cpp_docs.md`
- **Keyword Index**: `Unfold2d.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/cpu`):

- [`BinaryOpsKernel.cpp_docs.md_docs.md`](./BinaryOpsKernel.cpp_docs.md_docs.md)
- [`MultinomialKernel.cpp_kw.md_docs.md`](./MultinomialKernel.cpp_kw.md_docs.md)
- [`AmpGradScalerKernels.cpp_docs.md_docs.md`](./AmpGradScalerKernels.cpp_docs.md_docs.md)
- [`FusedSGDKernel.cpp_docs.md_docs.md`](./FusedSGDKernel.cpp_docs.md_docs.md)
- [`scaled_modified_bessel_k1.cpp_docs.md_docs.md`](./scaled_modified_bessel_k1.cpp_docs.md_docs.md)
- [`int_mm_kernel.h_docs.md_docs.md`](./int_mm_kernel.h_docs.md_docs.md)
- [`IsContiguous.h_docs.md_docs.md`](./IsContiguous.h_docs.md_docs.md)
- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`WeightNormKernel.cpp_kw.md_docs.md`](./WeightNormKernel.cpp_kw.md_docs.md)
- [`FusedAdamKernel.cpp_docs.md_docs.md`](./FusedAdamKernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Unfold2d.cpp_docs.md_docs.md`
- **Keyword Index**: `Unfold2d.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
