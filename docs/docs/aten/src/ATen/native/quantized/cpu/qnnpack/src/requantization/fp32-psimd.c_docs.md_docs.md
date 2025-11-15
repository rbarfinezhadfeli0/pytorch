# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-psimd.c_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-psimd.c_docs.md`
- **Size**: 6,849 bytes (6.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-psimd.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-psimd.c`
- **Size**: 4,345 bytes (4.24 KB)
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

#include <psimd.h>

#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_fp32__psimd(
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

  const psimd_f32 vscale = psimd_splat_f32(scale);
  const psimd_f32 vfmin = psimd_splat_f32(
      (float)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point));
  const psimd_f32 vfmax = psimd_splat_f32(
      (float)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point));
  const psimd_f32 vfmagic = psimd_splat_f32(12582912.0f);
  const psimd_s32 vimagic =
      psimd_splat_s32(INT32_C(0x4B400000) - (int32_t)(uint32_t)zero_point);
  for (; n != 0; n -= 16) {
    const psimd_s32 x = psimd_load_s32(input);
    const psimd_s32 y = psimd_load_s32(input + 4);
    const psimd_s32 z = psimd_load_s32(input + 8);
    const psimd_s32 w = psimd_load_s32(input + 12);
    input += 16;

    /*
     * Convert int32_t input to FP32 and multiply by FP32 scale.
     * Both operations involve roundings:
     * - Large int32_t values can't be exactly represented as FP32. We expect
     * that conversion instruction would round it to nearest FP32 value with
     * ties to even, but Clang documentation for __builtin_convertvector does
     *   not guarantee that.
     * - Product of two FP32 values is generally not exactly representation as
     * an FP32 value, and will be rounded to nearest FP32 value with ties to
     * even.
     */
    const psimd_f32 x_scaled = psimd_cvt_s32_f32(x) * vscale;
    const psimd_f32 y_scaled = psimd_cvt_s32_f32(y) * vscale;
    const psimd_f32 z_scaled = psimd_cvt_s32_f32(z) * vscale;
    const psimd_f32 w_scaled = psimd_cvt_s32_f32(w) * vscale;

    /*
     * Clang/gcc vector extension does not provide an intrinsics for a
     * floating-point to integer conversion operation with
     * rounding-to-nearest-even. In lieu of such intrinsic, we use a magic trick
     * of adding a large number (1.5 * 2**23) to scaled value to cause rounding
     * to integer, and then substracing this magic number as integer. This trick
     * works only in a limited range (absolute value of input must be less than
     * 2**22), so generally we have to clamp input to this range before using
     * the magic. However, clamping to any smaller range works just as well, and
     * thus we clamp to [qmin - zero point, qmax - zero point] range so that
     * after we add zero point to the result, it gets into target [qmin, qmax]
     * range.
     */
    const psimd_f32 x_clamped =
        psimd_min_f32(psimd_max_f32(x_scaled, vfmin), vfmax);
    const psimd_f32 y_clamped =
        psimd_min_f32(psimd_max_f32(y_scaled, vfmin), vfmax);
    const psimd_f32 z_clamped =
        psimd_min_f32(psimd_max_f32(z_scaled, vfmin), vfmax);
    const psimd_f32 w_clamped =
        psimd_min_f32(psimd_max_f32(w_scaled, vfmin), vfmax);

    /*
     * Conversion to integer using the "magic trick". Rounding is performed in
     * the output of addition operation, and result is rounded to nearest even
     * integer with ties to even.
     */
    const psimd_s32 x_biased = (psimd_s32)(x_clamped + vfmagic) - vimagic;
    const psimd_s32 y_biased = (psimd_s32)(y_clamped + vfmagic) - vimagic;
    const psimd_s32 z_biased = (psimd_s32)(z_clamped + vfmagic) - vimagic;
    const psimd_s32 w_biased = (psimd_s32)(w_clamped + vfmagic) - vimagic;

    /*
     * Select low 8 bits of each 32-bit integer in the vectors for the output.
     * Since result is already clamped to [qmin, qmax] subrange of [0, 255],
     * saturation is not needed.
     */
    const psimd_u16 xy_packed =
        psimd_concat_even_u16((psimd_u16)x_biased, (psimd_u16)y_biased);
    const psimd_u16 zw_packed =
        psimd_concat_even_u16((psimd_u16)z_biased, (psimd_u16)w_biased);

    const psimd_u8 xyzw_packed =
        psimd_concat_even_u8((psimd_u8)xy_packed, (psimd_u8)zw_packed);

    psimd_store_u8(output, xyzw_packed);
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

- **File Documentation**: `fp32-psimd.c_docs.md`
- **Keyword Index**: `fp32-psimd.c_kw.md`
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

- **File Documentation**: `fp32-psimd.c_docs.md_docs.md`
- **Keyword Index**: `fp32-psimd.c_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
