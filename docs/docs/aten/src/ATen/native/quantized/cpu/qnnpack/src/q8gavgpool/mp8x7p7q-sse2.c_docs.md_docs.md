# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool/mp8x7p7q-sse2.c_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool/mp8x7p7q-sse2.c_docs.md`
- **Size**: 15,751 bytes (15.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool/mp8x7p7q-sse2.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool/mp8x7p7q-sse2.c`
- **Size**: 13,544 bytes (13.23 KB)
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

#include <qnnpack/q8gavgpool.h>

void pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(m > 7);
  assert(n >= 8);

  const uint8_t* i0 = input;
  const uint8_t* i1 = i0 + input_stride;
  const uint8_t* i2 = i1 + input_stride;
  const uint8_t* i3 = i2 + input_stride;
  const uint8_t* i4 = i3 + input_stride;
  const uint8_t* i5 = i4 + input_stride;
  const uint8_t* i6 = i5 + input_stride;
  const size_t packed_n = (n + 7) & -8;
  const size_t input_increment = 7 * input_stride - packed_n;
  const __m128i vbias =
      _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);
  const __m128i vzero = _mm_setzero_si128();

  /* note: goes up to 7 elements over bound */
  int32_t* acc = buffer;
  for (size_t k = 0; k < n; k += 8) {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);
    i0 += 8;
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);
    i1 += 8;
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);
    i2 += 8;
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);
    i3 += 8;
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);
    i4 += 8;
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);
    i5 += 8;
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);
    i6 += 8;

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vxi0, vzero));
    __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    _mm_store_si128((__m128i*)acc, vacc_lo);
    _mm_store_si128((__m128i*)acc + 1, vacc_hi);
    acc += 8;
  }
  for (m -= 7; m > 7; m -= 7) {
    acc = buffer;
    i0 = (const uint8_t*)((uintptr_t)i0 + input_increment);
    i1 = (const uint8_t*)((uintptr_t)i1 + input_increment);
    i2 = (const uint8_t*)((uintptr_t)i2 + input_increment);
    i3 = (const uint8_t*)((uintptr_t)i3 + input_increment);
    i4 = (const uint8_t*)((uintptr_t)i4 + input_increment);
    i5 = (const uint8_t*)((uintptr_t)i5 + input_increment);
    i6 = (const uint8_t*)((uintptr_t)i6 + input_increment);

    /* note: goes up to 7 elements over bound */
    for (size_t k = 0; k < n; k += 8) {
      const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);
      i0 += 8;
      const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);
      i1 += 8;
      const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);
      i2 += 8;
      const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);
      i3 += 8;
      const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);
      i4 += 8;
      const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);
      i5 += 8;
      const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);
      i6 += 8;
      __m128i vacc_lo = _mm_load_si128((const __m128i*)acc);
      __m128i vacc_hi = _mm_load_si128((const __m128i*)acc + 1);

      const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
      const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
      const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
      const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
      const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
      const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
      const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

      _mm_store_si128((__m128i*)acc, vacc_lo);
      _mm_store_si128((__m128i*)acc + 1, vacc_hi);
      acc += 8;
    }
  }

  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

  i0 = (const uint8_t*)((uintptr_t)i0 + input_increment);
  i1 = (const uint8_t*)((uintptr_t)i1 + input_increment);
  if (m < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*)((uintptr_t)i2 + input_increment);
  if (m <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*)((uintptr_t)i3 + input_increment);
  if (m < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*)((uintptr_t)i4 + input_increment);
  if (m <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*)((uintptr_t)i5 + input_increment);
  if (m < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*)((uintptr_t)i6 + input_increment);
  if (m <= 6) {
    i6 = zero;
  }

  acc = buffer;
  do {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);
    i0 += 8;
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);
    i1 += 8;
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);
    i2 += 8;
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);
    i3 += 8;
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);
    i4 += 8;
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);
    i5 += 8;
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);
    i6 += 8;
    __m128i vacc_lo = _mm_load_si128((const __m128i*)acc);
    __m128i vacc_hi = _mm_load_si128((const __m128i*)acc + 1);
    acc += 8;

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

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

    _mm_storel_epi64((__m128i*)output, vout);
    output += 8;

    n -= 8;
  } while (n >= 8);
  if (n != 0) {
    const size_t address_decrement = 8 - n;
    i0 = (const uint8_t*)((uintptr_t)i0 - address_decrement);
    i1 = (const uint8_t*)((uintptr_t)i1 - address_decrement);
    i2 = (const uint8_t*)((uintptr_t)i2 - address_decrement);
    i3 = (const uint8_t*)((uintptr_t)i3 - address_decrement);
    i4 = (const uint8_t*)((uintptr_t)i4 - address_decrement);
    i5 = (const uint8_t*)((uintptr_t)i5 - address_decrement);
    i6 = (const uint8_t*)((uintptr_t)i6 - address_decrement);
    const __m128i vi_shift = _mm_cvtsi32_si128(8 * address_decrement);

    const __m128i vi0 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i0), vi_shift);
    const __m128i vi1 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i1), vi_shift);
    const __m128i vi2 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i2), vi_shift);
    const __m128i vi3 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i3), vi_shift);
    const __m128i vi4 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i4), vi_shift);
    const __m128i vi5 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i5), vi_shift);
    const __m128i vi6 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i6), vi_shift);
    __m128i vacc_lo = _mm_load_si128((const __m128i*)acc);
    __m128i vacc_hi = _mm_load_si128((const __m128i*)acc + 1);

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

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

    if (n & 4) {
      *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
      output += 4;
      vout = _mm_srli_epi64(vout, 32);
    }
    if (n & 2) {
      *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
      output += 2;
      vout = _mm_srli_epi32(vout, 16);
    }
    if (n & 1) {
      *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
    }
  }
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool`):

- [`mp8x7p7q-neon.c_docs.md`](./mp8x7p7q-neon.c_docs.md)
- [`up8x7-neon.c_docs.md`](./up8x7-neon.c_docs.md)
- [`up8xm-sse2.c_docs.md`](./up8xm-sse2.c_docs.md)
- [`up8x7-sse2.c_docs.md`](./up8x7-sse2.c_docs.md)
- [`up8xm-neon.c_docs.md`](./up8xm-neon.c_docs.md)


## Cross-References

- **File Documentation**: `mp8x7p7q-sse2.c_docs.md`
- **Keyword Index**: `mp8x7p7q-sse2.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool`):

- [`up8x7-sse2.c_kw.md_docs.md`](./up8x7-sse2.c_kw.md_docs.md)
- [`up8xm-sse2.c_docs.md_docs.md`](./up8xm-sse2.c_docs.md_docs.md)
- [`up8xm-neon.c_kw.md_docs.md`](./up8xm-neon.c_kw.md_docs.md)
- [`up8xm-sse2.c_kw.md_docs.md`](./up8xm-sse2.c_kw.md_docs.md)
- [`mp8x7p7q-neon.c_kw.md_docs.md`](./mp8x7p7q-neon.c_kw.md_docs.md)
- [`up8x7-neon.c_kw.md_docs.md`](./up8x7-neon.c_kw.md_docs.md)
- [`up8x7-neon.c_docs.md_docs.md`](./up8x7-neon.c_docs.md_docs.md)
- [`mp8x7p7q-neon.c_docs.md_docs.md`](./mp8x7p7q-neon.c_docs.md_docs.md)
- [`mp8x7p7q-sse2.c_kw.md_docs.md`](./mp8x7p7q-sse2.c_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mp8x7p7q-sse2.c_docs.md_docs.md`
- **Keyword Index**: `mp8x7p7q-sse2.c_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
