# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8avgpool/up8xm-sse2.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8avgpool/up8xm-sse2.c`
- **Size**: 3,032 bytes (2.96 KB)
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

#include <emmintrin.h>

#include <qnnpack/q8avgpool.h>

void pytorch_q8avgpool_ukernel_up8xm__sse2(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(n != 0);
  assert(ks != 0);
  assert(kc < 8);

  const __m128i vbias =
      _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);
  const __m128i vzero = _mm_setzero_si128();
  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

  do {
    const uint8_t** next_input =
        (const uint8_t**)((uintptr_t)input + input_increment);
    __m128i vacc_lo = vbias;
    __m128i vacc_hi = vbias;

    size_t m = ks;
    do {
      const uint8_t* i = *input++;
      i += kc;
      __m128i vi = _mm_setzero_si128();
      if (kc & 1) {
        i -= 1;
        vi = _mm_cvtsi32_si128((int)(uint32_t)*i);
      }
      if (kc & 2) {
        vi = _mm_slli_epi32(vi, 16);
        i -= 2;
        vi = _mm_insert_epi16(vi, *((const uint16_t*)i), 0);
      }
      if (kc & 4) {
        i -= 4;
        vi = _mm_unpacklo_epi32(
            _mm_cvtsi32_si128((int)*((const uint32_t*)i)), vi);
      }

      const __m128i vxi = _mm_unpacklo_epi8(vi, vzero);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi, vzero));
    } while (--m != 0);
    input = next_input;

    const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
    const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

    const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
    const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

    __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
    vout = _mm_adds_epi16(
        vout,
        _mm_load_si128(
            (const __m128i*)quantization_params->sse2.output_zero_point));
    vout = _mm_packus_epi16(vout, vout);
    vout = _mm_min_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
    vout = _mm_max_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

    if (kc & 4) {
      *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
      output += 4;
      vout = _mm_srli_epi64(vout, 32);
    }
    if (kc & 2) {
      *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
      output += 2;
      vout = _mm_srli_epi32(vout, 16);
    }
    if (kc & 1) {
      *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
      output += 1;
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8avgpool`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8avgpool`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src/q8avgpool`):

- [`mp8x9p8q-sse2.c_docs.md`](./mp8x9p8q-sse2.c_docs.md)
- [`up8x9-neon.c_docs.md`](./up8x9-neon.c_docs.md)
- [`up8x9-sse2.c_docs.md`](./up8x9-sse2.c_docs.md)
- [`mp8x9p8q-neon.c_docs.md`](./mp8x9p8q-neon.c_docs.md)
- [`up8xm-neon.c_docs.md`](./up8xm-neon.c_docs.md)


## Cross-References

- **File Documentation**: `up8xm-sse2.c_docs.md`
- **Keyword Index**: `up8xm-sse2.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
