# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-sse2.c_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-sse2.c_docs.md`
- **Size**: 5,092 bytes (4.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-sse2.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-sse2.c`
- **Size**: 2,633 bytes (2.57 KB)
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

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

#include "gemmlowp-sse.h"

void pytorch_qnnp_requantize_gemmlowp__sse2(
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

  /* Compute requantization parameters */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);
  const __m128i vqmin = _mm_set1_epi8((char)qmin);
  const __m128i vqmax = _mm_set1_epi8((char)qmax);
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    const __m128i x_product = gemmlowp_sse_vqrdmulh_s32(x, vmultiplier);
    const __m128i y_product = gemmlowp_sse_vqrdmulh_s32(y, vmultiplier);
    const __m128i z_product = gemmlowp_sse_vqrdmulh_s32(z, vmultiplier);
    const __m128i w_product = gemmlowp_sse_vqrdmulh_s32(w, vmultiplier);

    const __m128i x_scaled = gemmlowp_sse_rdivbypo2_s32(x_product, shift);
    const __m128i y_scaled = gemmlowp_sse_rdivbypo2_s32(y_product, shift);
    const __m128i z_scaled = gemmlowp_sse_rdivbypo2_s32(z_product, shift);
    const __m128i w_scaled = gemmlowp_sse_rdivbypo2_s32(w_product, shift);

    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

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

- **File Documentation**: `gemmlowp-sse2.c_docs.md`
- **Keyword Index**: `gemmlowp-sse2.c_kw.md`
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

- **File Documentation**: `gemmlowp-sse2.c_docs.md_docs.md`
- **Keyword Index**: `gemmlowp-sse2.c_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
