# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-scalar.c_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-scalar.c_docs.md`
- **Size**: 6,539 bytes (6.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-scalar.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-scalar.c`
- **Size**: 4,031 bytes (3.94 KB)
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
#include <math.h>
#include <stdint.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_fp32__scalar_lrintf(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 4 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const long lmin =
      (long)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point);
  const long lmax =
      (long)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point);
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    const float x_scaled = (float)x * scale;
    const float y_scaled = (float)y * scale;
    const float z_scaled = (float)z * scale;
    const float w_scaled = (float)w * scale;

    const long x_rounded = lrintf(x_scaled);
    const long y_rounded = lrintf(y_scaled);
    const long z_rounded = lrintf(z_scaled);
    const long w_rounded = lrintf(w_scaled);

    const int32_t x_clamped = (int32_t)(
        x_rounded < lmin ? lmin : x_rounded > lmax ? lmax : x_rounded);
    const int32_t y_clamped = (int32_t)(
        y_rounded < lmin ? lmin : y_rounded > lmax ? lmax : y_rounded);
    const int32_t z_clamped = (int32_t)(
        z_rounded < lmin ? lmin : z_rounded > lmax ? lmax : z_rounded);
    const int32_t w_clamped = (int32_t)(
        w_rounded < lmin ? lmin : w_rounded > lmax ? lmax : w_rounded);

    const int32_t x_biased = x_clamped + (int32_t)(uint32_t)zero_point;
    const int32_t y_biased = y_clamped + (int32_t)(uint32_t)zero_point;
    const int32_t z_biased = z_clamped + (int32_t)(uint32_t)zero_point;
    const int32_t w_biased = w_clamped + (int32_t)(uint32_t)zero_point;

    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    output += 4;
  }
}

void pytorch_qnnp_requantize_fp32__scalar_magic(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 4 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const float fmin =
      (float)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point);
  const float fmax =
      (float)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point);
  const float fmagic = 12582912.0f;
  const int32_t imagic = INT32_C(0x4B400000) - (int32_t)(uint32_t)zero_point;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    const float x_scaled = (float)x * scale;
    const float y_scaled = (float)y * scale;
    const float z_scaled = (float)z * scale;
    const float w_scaled = (float)w * scale;

    const float x_clamped =
        x_scaled < fmin ? fmin : x_scaled > fmax ? fmax : x_scaled;
    const float y_clamped =
        y_scaled < fmin ? fmin : y_scaled > fmax ? fmax : y_scaled;
    const float z_clamped =
        z_scaled < fmin ? fmin : z_scaled > fmax ? fmax : z_scaled;
    const float w_clamped =
        w_scaled < fmin ? fmin : w_scaled > fmax ? fmax : w_scaled;

    const int32_t x_biased = (int32_t)fp32_to_bits(x_clamped + fmagic) - imagic;
    const int32_t y_biased = (int32_t)fp32_to_bits(y_clamped + fmagic) - imagic;
    const int32_t z_biased = (int32_t)fp32_to_bits(z_clamped + fmagic) - imagic;
    const int32_t w_biased = (int32_t)fp32_to_bits(w_clamped + fmagic) - imagic;

    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    output += 4;
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

- **File Documentation**: `fp32-scalar.c_docs.md`
- **Keyword Index**: `fp32-scalar.c_kw.md`
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

- **File Documentation**: `fp32-scalar.c_docs.md_docs.md`
- **Keyword Index**: `fp32-scalar.c_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
