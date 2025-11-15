# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm/4x4c2-dq-sse2.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm/4x4c2-dq-sse2.c`
- **Size**: 11,857 bytes (11.58 KB)
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

#include <immintrin.h>

#include <qnnpack/q8gemm.h>
#include <requantization/runtime-sse2.h>

void pytorch_q8gemm_dq_ukernel_4x4c2__sse2(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    const float* restrict b,
    float* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const struct pytorch_qnnp_conv_dynamic_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  __m128i vacc0x0123 = _mm_setzero_si128();
  __m128i vacc1x0123 = _mm_setzero_si128();
  __m128i vacc2x0123 = _mm_setzero_si128();
  __m128i vacc3x0123 = _mm_setzero_si128();
  w = (const void*)((uintptr_t)w + 16);

  const uint8_t* a0 = a;
  const uint8_t* a1 = (const uint8_t*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const uint8_t* a2 = (const uint8_t*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const uint8_t* a3 = (const uint8_t*)((uintptr_t)a2 + a_stride);
  if (mr != 4) {
    a3 = a2;
  }

  const __m128i va_zero_point = _mm_set1_epi16(quantization_params->input_zero_point);
  const int16_t vb_zero_point_0 =
    (int16_t)(uint16_t)quantization_params->kernel_zero_points[
    output_channel_index];
  const int16_t vb_zero_point_1 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 1];
  const int16_t vb_zero_point_2 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 2];
  const int16_t vb_zero_point_3 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 3];

  __m128i vb_zero_point = _mm_set_epi16(vb_zero_point_3,
                                        vb_zero_point_3,
                                        vb_zero_point_2,
                                        vb_zero_point_2,
                                        vb_zero_point_1,
                                        vb_zero_point_1,
                                        vb_zero_point_0,
                                        vb_zero_point_0
                                        );
  const __m128 vmultiplier =
      _mm_loadu_ps(&quantization_params->multipliers[output_channel_index]);

  const __m128 vbias = _mm_load_ps(b);

  const __m128i vzero = _mm_setzero_si128();
  for (; k >= 8; k -= 8) {
    const __m128i va0 = _mm_loadl_epi64((const __m128i*)a0);
    const __m128i vxa0 =
        sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
    a0 += 8;
    const __m128i va1 = _mm_loadl_epi64((const __m128i*)a1);
    const __m128i vxa1 =
        sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
    a1 += 8;
    const __m128i va2 = _mm_loadl_epi64((const __m128i*)a2);
    const __m128i vxa2 =
        sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
    a2 += 8;
    const __m128i va3 = _mm_loadl_epi64((const __m128i*)a3);
    const __m128i vxa3 =
        sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
    a3 += 8;

    const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
    const __m128i vxb0 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    vacc3x0123 = _mm_add_epi32(

        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));

    const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
    const __m128i vxb1 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

    const __m128i vb2 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
    const __m128i vxb2 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

    const __m128i vb3 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
    const __m128i vxb3 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 32);

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
  }
  if (k != 0) {
    const size_t a_predecrement = 8 - k;
    const __m128i va_shift = _mm_cvtsi32_si128(8 * a_predecrement);

    const __m128i va0 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a0 - a_predecrement)), va_shift);
    const __m128i vxa0 =
        sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
    const __m128i va1 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a1 - a_predecrement)), va_shift);
    const __m128i vxa1 =
        sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
    const __m128i va2 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a2 - a_predecrement)), va_shift);
    const __m128i vxa2 =
        sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
    const __m128i va3 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a3 - a_predecrement)), va_shift);
    const __m128i vxa3 =
        sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);

    const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
    const __m128i vxb0 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));

    if (k > 2) {
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
      const __m128i vxb1 =
          _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

      vacc0x0123 = _mm_add_epi32(
          vacc0x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      vacc1x0123 = _mm_add_epi32(
          vacc1x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      vacc2x0123 = _mm_add_epi32(
          vacc2x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      vacc3x0123 = _mm_add_epi32(
          vacc3x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

      if (k > 4) {
        const __m128i vb2 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
        const __m128i vxb2 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

        if (k > 6) {
          const __m128i vb3 =
              _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
          const __m128i vxb3 =
              _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);

          vacc0x0123 = _mm_add_epi32(
              vacc0x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
          vacc1x0123 = _mm_add_epi32(
              vacc1x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
          vacc2x0123 = _mm_add_epi32(
              vacc2x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
          vacc3x0123 = _mm_add_epi32(
              vacc3x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
        }
      }
    }
  }

  __m128 vout0 = _mm_mul_ps(vmultiplier, _mm_cvtepi32_ps(vacc0x0123));
  __m128 vout1 = _mm_mul_ps(vmultiplier, _mm_cvtepi32_ps(vacc1x0123));
  __m128 vout2 = _mm_mul_ps(vmultiplier, _mm_cvtepi32_ps(vacc2x0123));
  __m128 vout3 = _mm_mul_ps(vmultiplier, _mm_cvtepi32_ps(vacc3x0123));

  vout0 = _mm_add_ps(vout0, vbias);
  vout1 = _mm_add_ps(vout1, vbias);
  vout2 = _mm_add_ps(vout2, vbias);
  vout3 = _mm_add_ps(vout3, vbias);

  float* c0 = c;
  float* c1 = c0 + c_stride;
  if (mr < 2) {
    c1 = c0;
  }
  float* c2 = c1 + c_stride;
  if (mr <= 2) {
    c2 = c1;
  }
  float* c3 = c2 + c_stride;
  if (mr != 4) {
    c3 = c2;
  }

  if (nr == 4) {
    _mm_storeu_ps(c0, vout0);
    _mm_storeu_ps(c1, vout1);
    _mm_storeu_ps(c2, vout2);
    _mm_storeu_ps(c3, vout3);
  } else {
    if (nr >= 2) {
      _mm_storel_pi((__m64*)c0, vout0);
      _mm_storel_pi((__m64*)c1, vout1);
      _mm_storel_pi((__m64*)c2, vout2);
      _mm_storel_pi((__m64*)c3, vout3);

      c0 += 2;
      vout0 = _mm_shuffle_ps(vout0, vout0, _MM_SHUFFLE(2, 2, 2, 2));
      c1 += 2;
      vout1 = _mm_shuffle_ps(vout1, vout1, _MM_SHUFFLE(2, 2, 2, 2));
      c2 += 2;
      vout2 = _mm_shuffle_ps(vout2, vout2, _MM_SHUFFLE(2, 2, 2, 2));
      c3 += 2;
      vout3 = _mm_shuffle_ps(vout3, vout3, _MM_SHUFFLE(2, 2, 2, 2));

      nr -= 2;
    }
    if (nr != 0) {
      *c0 = _mm_cvtss_f32(vout0);
      *c1 = _mm_cvtss_f32(vout1);
      *c2 = _mm_cvtss_f32(vout2);
      *c3 = _mm_cvtss_f32(vout3);
    }
  }
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm`):

- [`8x8-neon.c_docs.md`](./8x8-neon.c_docs.md)
- [`2x4c8-sse2.c_docs.md`](./2x4c8-sse2.c_docs.md)
- [`4x4c2-sse2.c_docs.md`](./4x4c2-sse2.c_docs.md)
- [`4x8c2-xzp-neon.c_docs.md`](./4x8c2-xzp-neon.c_docs.md)
- [`4x8-dq-neon.c_docs.md`](./4x8-dq-neon.c_docs.md)
- [`6x4-neon.c_docs.md`](./6x4-neon.c_docs.md)
- [`4x8-neon.c_docs.md`](./4x8-neon.c_docs.md)
- [`4x-sumrows-neon.c_docs.md`](./4x-sumrows-neon.c_docs.md)


## Cross-References

- **File Documentation**: `4x4c2-dq-sse2.c_docs.md`
- **Keyword Index**: `4x4c2-dq-sse2.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
