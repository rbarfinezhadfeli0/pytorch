# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-sse2.c_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-sse2.c_docs.md`
- **Size**: 6,669 bytes (6.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-sse2.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-sse2.c`
- **Size**: 4,169 bytes (4.07 KB)
- **Type**: Source File (.c)
- **Extension**: `.c`

## File Purpose

This is a source file (.c) that is part of the PyTorch project.

## Original Source

```c
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <emmintrin.h>

#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_fp32__sse2(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 16 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const __m128 vscale = _mm_set1_ps(scale);
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);
  const __m128i vqmin = _mm_set1_epi8((char)qmin);
  const __m128i vqmax = _mm_set1_epi8((char)qmax);
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    /*
     * Convert int32_t input to FP32 and multiply by FP32 scale.
     * Both operations involve statistically unbiased roundings (with default
     * MXCSR rounding mode):
     * - Large int32_t values can't be exactly represented as FP32. CVTDQ2PS
     * instruction on x86 would round it according to nearest FP32 value with
     * ties to even (assuming default MXCSR rounding mode).
     * - Product of two FP32 values is generally not exactly representation as
     * an FP32 value, and will be rounded to nearest FP32 value with ties to
     * even with default MXCSR rounding mode.
     */
    const __m128 x_scaled = _mm_mul_ps(_mm_cvtepi32_ps(x), vscale);
    const __m128 y_scaled = _mm_mul_ps(_mm_cvtepi32_ps(y), vscale);
    const __m128 z_scaled = _mm_mul_ps(_mm_cvtepi32_ps(z), vscale);
    const __m128 w_scaled = _mm_mul_ps(_mm_cvtepi32_ps(w), vscale);

    /*
     * Convert scaled FP32 result to int32_t using CVTPS2DQ instruction from x86
     * SSE2. CVTPS2DQ instruction rounds result according to nearest FP32 value
     * with ties to even (assuming default MXCSR rounding mode). However, when
     * conversion overflows, it produces INT32_MIN as a result. For large
     * positive inputs the result of conversion can become negative, which
     * affects the final requantization result. Note that on x86 SSE2 we have
     * e.g. int32_t(float(INT32_MAX)) == INT32_MIN! This happens because
     * float(INT32_MAX) rounds to 2**31, which overflows int32_t when it is
     * converted back to integer.
     *
     * Thankfully, we can prove that overflow never happens in this
     * requantization scheme. The largest positive input is INT32_MAX (2**31 -
     * 1), which turns into 2**31 when converted to float. The largest scale
     * value is 0x1.FFFFFEp-1. When multiplied together, the result is
     * 2147483520 (compare to INT32_MAX = 2147483647), which fits into int32_t
     * without overflow.
     */
    const __m128i x_rounded = _mm_cvtps_epi32(x_scaled);
    const __m128i y_rounded = _mm_cvtps_epi32(y_scaled);
    const __m128i z_rounded = _mm_cvtps_epi32(z_scaled);
    const __m128i w_rounded = _mm_cvtps_epi32(w_scaled);

    /*
     * Standard final sequence on x86 SSE2:
     * - Pack to int16_t and saturate
     * - Add zero point
     * - Pack to uint8_t and saturate
     * - Clamp between qmin and qmax
     */
    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_rounded, y_rounded), vzero_point);
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_rounded, w_rounded), vzero_point);
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

    /*
     * 4x CVTDQ2PS
     * 4x MULPS
     * 4x CVTPS2DQ
     * 2x PACKSSDW
     * 1x PACKUSWB
     * 2x PADDW
     * 1x PMAXUB
     * 1x PMINUB
     * ---------------------
     * 19 instructions total
     */

    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    output += 16;
  }
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization`):

- [`gemmlowp-sse2.c_docs.md`](./gemmlowp-sse2.c_docs.md)
- [`precise-sse2.c_docs.md`](./precise-sse2.c_docs.md)
- [`runtime-sse2.h_docs.md`](./runtime-sse2.h_docs.md)
- [`q31-sse4.c_docs.md`](./q31-sse4.c_docs.md)
- [`fp32-neon.c_docs.md`](./fp32-neon.c_docs.md)
- [`gemmlowp-neon.c_docs.md`](./gemmlowp-neon.c_docs.md)
- [`gemmlowp-ssse3.c_docs.md`](./gemmlowp-ssse3.c_docs.md)
- [`precise-scalar.c_docs.md`](./precise-scalar.c_docs.md)
- [`precise-psimd.c_docs.md`](./precise-psimd.c_docs.md)
- [`gemmlowp-sse4.c_docs.md`](./gemmlowp-sse4.c_docs.md)


## Cross-References

- **File Documentation**: `fp32-sse2.c_docs.md`
- **Keyword Index**: `fp32-sse2.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization`):

- [`runtime-assembly.h_docs.md_docs.md`](./runtime-assembly.h_docs.md_docs.md)
- [`precise-psimd.c_kw.md_docs.md`](./precise-psimd.c_kw.md_docs.md)
- [`fp32-neon.c_docs.md_docs.md`](./fp32-neon.c_docs.md_docs.md)
- [`gemmlowp-scalar.c_kw.md_docs.md`](./gemmlowp-scalar.c_kw.md_docs.md)
- [`precise-scalar.c_kw.md_docs.md`](./precise-scalar.c_kw.md_docs.md)
- [`q31-sse4.c_docs.md_docs.md`](./q31-sse4.c_docs.md_docs.md)
- [`q31-ssse3.c_kw.md_docs.md`](./q31-ssse3.c_kw.md_docs.md)
- [`precise-sse2.c_kw.md_docs.md`](./precise-sse2.c_kw.md_docs.md)
- [`gemmlowp-sse.h_kw.md_docs.md`](./gemmlowp-sse.h_kw.md_docs.md)
- [`gemmlowp-neon.c_docs.md_docs.md`](./gemmlowp-neon.c_docs.md_docs.md)


## Cross-References

- **File Documentation**: `fp32-sse2.c_docs.md_docs.md`
- **Keyword Index**: `fp32-sse2.c_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
