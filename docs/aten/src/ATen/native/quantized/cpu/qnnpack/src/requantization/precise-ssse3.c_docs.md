# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/precise-ssse3.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/precise-ssse3.c`
- **Size**: 5,696 bytes (5.56 KB)
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

#include <tmmintrin.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_precise__ssse3(
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

  const uint32_t scale_bits = fp32_to_bits(scale);
  const uint32_t multiplier =
      (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);
  assert(shift < 56);
  const uint64_t rounding = UINT64_C(1) << (shift - 1);

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);
  const __m128i vqmin = _mm_set1_epi8((char)qmin);
  const __m128i vqmax = _mm_set1_epi8((char)qmax);
  const __m128i vshift = _mm_cvtsi32_si128((int)shift);
  const __m128i vrounding = _mm_set1_epi64x(rounding);
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    const __m128i x_abs0123 = _mm_abs_epi32(x);
    const __m128i y_abs0123 = _mm_abs_epi32(y);
    const __m128i z_abs0123 = _mm_abs_epi32(z);
    const __m128i w_abs0123 = _mm_abs_epi32(w);

    const __m128i x_abs1032 =
        _mm_shuffle_epi32(x_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i y_abs1032 =
        _mm_shuffle_epi32(y_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i z_abs1032 =
        _mm_shuffle_epi32(z_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i w_abs1032 =
        _mm_shuffle_epi32(w_abs0123, _MM_SHUFFLE(2, 3, 0, 1));

    const __m128i x_absmul02 = _mm_mul_epu32(x_abs0123, vmultiplier);
    const __m128i y_absmul02 = _mm_mul_epu32(y_abs0123, vmultiplier);
    const __m128i z_absmul02 = _mm_mul_epu32(z_abs0123, vmultiplier);
    const __m128i w_absmul02 = _mm_mul_epu32(w_abs0123, vmultiplier);

    const __m128i x_absmul13 = _mm_mul_epu32(x_abs1032, vmultiplier);
    const __m128i y_absmul13 = _mm_mul_epu32(y_abs1032, vmultiplier);
    const __m128i z_absmul13 = _mm_mul_epu32(z_abs1032, vmultiplier);
    const __m128i w_absmul13 = _mm_mul_epu32(w_abs1032, vmultiplier);

    const __m128i x_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(x_absmul02, vrounding), vshift);
    const __m128i x_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(x_absmul13, vrounding), vshift);
    const __m128i y_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(y_absmul02, vrounding), vshift);
    const __m128i y_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(y_absmul13, vrounding), vshift);
    const __m128i z_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(z_absmul02, vrounding), vshift);
    const __m128i z_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(z_absmul13, vrounding), vshift);
    const __m128i w_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(w_absmul02, vrounding), vshift);
    const __m128i w_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(w_absmul13, vrounding), vshift);

    const __m128i x_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x_abs_scaled02),
        _mm_castsi128_ps(x_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i y_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(y_abs_scaled02),
        _mm_castsi128_ps(y_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i z_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(z_abs_scaled02),
        _mm_castsi128_ps(z_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i w_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(w_abs_scaled02),
        _mm_castsi128_ps(w_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));

    const __m128i x_abs_scaled =
        _mm_shuffle_epi32(x_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i y_abs_scaled =
        _mm_shuffle_epi32(y_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i z_abs_scaled =
        _mm_shuffle_epi32(z_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i w_abs_scaled =
        _mm_shuffle_epi32(w_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i x_scaled = _mm_sign_epi32(x_abs_scaled, x);
    const __m128i y_scaled = _mm_sign_epi32(y_abs_scaled, y);
    const __m128i z_scaled = _mm_sign_epi32(z_abs_scaled, z);
    const __m128i w_scaled = _mm_sign_epi32(w_abs_scaled, w);

    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

    /*
     * 4x PABSD
     * 8x PSHUFD
     * 8x PMULUDQ
     * 8x PSRLQ
     * 8x PADDQ
     * 4x SHUFPS
     * 4x PSIGND
     * 2x PACKSSDW
     * 1x PACKUSWB
     * 2x PADDW
     * 1x PMAXUB
     * 1x PMINUB
     * ---------------------
     * 51 instructions total
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

- **File Documentation**: `precise-ssse3.c_docs.md`
- **Keyword Index**: `precise-ssse3.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
