# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/scalar-utils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/scalar-utils.h`
- **Size**: 3,517 bytes (3.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#include <fp16/bitcasts.h>

#if defined(__clang__)
#if __clang_major__ == 3 && __clang_minor__ >= 7 || __clang_major__ > 3
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB \
  __attribute__((__no_sanitize__("shift-base")))
#else
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
#endif
#elif defined(__GNUC__)
#if __GNUC__ >= 8
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB \
  __attribute__((__no_sanitize__("shift-base")))
#elif __GNUC__ == 4 && __GNUC_MINOR__ >= 9 || __GNUC__ > 4
/* 4.9 <= gcc < 8 support ubsan, but doesn't support no_sanitize attribute */
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
#ifndef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
#define PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND 1
#endif
#else
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
#endif
#else
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
#endif

PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
inline static int32_t asr_s32(int32_t x, uint32_t n) {
#ifdef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
#if defined(__x86_64__) || defined(__aarch64__)
  return (int32_t)((uint64_t)(int64_t)x >> n);
#else
  return x >= 0 ? x >> n : ~(~x >> n);
#endif
#else
  return x >> n;
#endif
}

PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
inline static int64_t asr_s64(int64_t x, uint32_t n) {
#ifdef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
  return x >= 0 ? x >> n : ~(~x >> n);
#else
  return x >> n;
#endif
}

inline static uint8_t pytorch_scalar_requantize_precise(
    int32_t value,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax) {
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = fp32_to_bits(scale);
  const uint32_t multiplier =
      (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);
  assert(shift < 56);

  /*
   * Compute absolute value of input as unsigned 32-bit int.
   * All further computations will work with unsigned values to avoid undefined
   * behaviour on signed operations.
   */
  const uint32_t abs_value = (value >= 0) ? (uint32_t)value : -(uint32_t)value;

  /* Compute full 64-bit product of 32-bit factors */
  const uint64_t product = (uint64_t)abs_value * (uint64_t)multiplier;

  /*
   * Shift the full 64-bit product right with rounding.
   * Rounding is performed towards closest integer, with midpoints rounded up
   * (same as away from zero).
   */
  const uint64_t rounding = UINT64_C(1) << (shift - 1);
  const uint32_t abs_scaled_value = (uint32_t)((product + rounding) >> shift);

  /*
   * Copy the sign of input to scaled absolute input value.
   */
  const int32_t scaled_value =
      (int32_t)(value >= 0 ? abs_scaled_value : -abs_scaled_value);

  /* Clamp scaled value with zero point between smin and smax */
  int32_t clamped_value = scaled_value;
  const int32_t smin = (int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point;
  if (clamped_value < smin) {
    clamped_value = smin;
  }
  const int32_t smax = (int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point;
  if (clamped_value > smax) {
    clamped_value = smax;
  }

  /* Add zero point to clamped value */
  const int32_t biased_value = clamped_value + (int32_t)(uint32_t)zero_point;

  return biased_value;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `assert.h`
- `limits.h`
- `stdbool.h`
- `stdint.h`
- `fp16/bitcasts.h`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack`):

- [`log.h_docs.md`](./log.h_docs.md)
- [`x8zip.h_docs.md`](./x8zip.h_docs.md)
- [`AlignedAllocator.h_docs.md`](./AlignedAllocator.h_docs.md)
- [`requantization.h_docs.md`](./requantization.h_docs.md)
- [`pack.h_docs.md`](./pack.h_docs.md)
- [`common.h_docs.md`](./common.h_docs.md)
- [`u8maxpool.h_docs.md`](./u8maxpool.h_docs.md)
- [`assembly.h_docs.md`](./assembly.h_docs.md)
- [`q8gavgpool.h_docs.md`](./q8gavgpool.h_docs.md)
- [`sdwconv.h_docs.md`](./sdwconv.h_docs.md)


## Cross-References

- **File Documentation**: `scalar-utils.h_docs.md`
- **Keyword Index**: `scalar-utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
