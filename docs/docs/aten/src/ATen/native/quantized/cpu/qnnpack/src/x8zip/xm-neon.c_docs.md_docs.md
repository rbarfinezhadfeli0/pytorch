# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/xm-neon.c_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/xm-neon.c_docs.md`
- **Size**: 8,235 bytes (8.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/xm-neon.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/xm-neon.c`
- **Size**: 6,025 bytes (5.88 KB)
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

#include <arm_neon.h>

#include <qnnpack/x8zip.h>

void pytorch_qnnp_x8zip_xm__neon(
    size_t n,
    size_t m,
    const void* input,
    void* output) {
  const uint8_t* w = input;
  const size_t input_increment = n * 3;
  const size_t output_increment = 4 - m * n;
  const uint8_t* last_input = w + n * (m - 1);
  void* last_output = (void*)((uintptr_t)output + (m - 4));

  if (n >= 8) {
    for (size_t i = 0; i < m; i += 4) {
      size_t k = n;
      w = (const uint8_t*)((uintptr_t)w + input_increment);
      if (w >= last_input) {
        w = last_input;
      }
      const uint8_t* z = (const uint8_t*)((uintptr_t)w - n);
      const uint8_t* y = (const uint8_t*)((uintptr_t)z - n);
      const uint8_t* x = (const uint8_t*)((uintptr_t)y - n);
      while (k >= 8) {
        const uint8x8_t vx = vld1_u8(x);
        x += 8;
        const uint8x8_t vy = vld1_u8(y);
        y += 8;
        const uint8x8_t vz = vld1_u8(z);
        z += 8;
        const uint8x8_t vw = vld1_u8(w);
        w += 8;

        const uint8x8x2_t vxy = vzip_u8(vx, vy);
        const uint8x8x2_t vzw = vzip_u8(vz, vw);
        const uint16x4x2_t vxyzw_lo = vzip_u16(
            vreinterpret_u16_u8(vxy.val[0]), vreinterpret_u16_u8(vzw.val[0]));
        const uint16x4x2_t vxyzw_hi = vzip_u16(
            vreinterpret_u16_u8(vxy.val[1]), vreinterpret_u16_u8(vzw.val[1]));

        vst1_lane_u32(
            __builtin_assume_aligned(output, 1),
            vreinterpret_u32_u16(vxyzw_lo.val[0]),
            0);
        output = (void*)((uintptr_t)output + m);

        vst1_lane_u32(
            __builtin_assume_aligned(output, 1),
            vreinterpret_u32_u16(vxyzw_lo.val[0]),
            1);
        output = (void*)((uintptr_t)output + m);

        vst1_lane_u32(
            __builtin_assume_aligned(output, 1),
            vreinterpret_u32_u16(vxyzw_lo.val[1]),
            0);
        output = (void*)((uintptr_t)output + m);

        vst1_lane_u32(
            __builtin_assume_aligned(output, 1),
            vreinterpret_u32_u16(vxyzw_lo.val[1]),
            1);
        output = (void*)((uintptr_t)output + m);

        vst1_lane_u32(
            __builtin_assume_aligned(output, 1),
            vreinterpret_u32_u16(vxyzw_hi.val[0]),
            0);
        output = (void*)((uintptr_t)output + m);

        vst1_lane_u32(
            __builtin_assume_aligned(output, 1),
            vreinterpret_u32_u16(vxyzw_hi.val[0]),
            1);
        output = (void*)((uintptr_t)output + m);

        vst1_lane_u32(
            __builtin_assume_aligned(output, 1),
            vreinterpret_u32_u16(vxyzw_hi.val[1]),
            0);
        output = (void*)((uintptr_t)output + m);

        vst1_lane_u32(
            __builtin_assume_aligned(output, 1),
            vreinterpret_u32_u16(vxyzw_hi.val[1]),
            1);
        output = (void*)((uintptr_t)output + m);

        k -= 8;
      }
      if (k != 0) {
        const size_t address_increment = k - 8;
        x = (const uint8_t*)((uintptr_t)x + address_increment);
        y = (const uint8_t*)((uintptr_t)y + address_increment);
        z = (const uint8_t*)((uintptr_t)z + address_increment);
        w = (const uint8_t*)((uintptr_t)w + address_increment);
        const int64x1_t vshift = vmov_n_s64(8 * address_increment);

        const uint64x1_t vx = vshl_u64(vreinterpret_u64_u8(vld1_u8(x)), vshift);
        const uint64x1_t vy = vshl_u64(vreinterpret_u64_u8(vld1_u8(y)), vshift);
        const uint64x1_t vz = vshl_u64(vreinterpret_u64_u8(vld1_u8(z)), vshift);
        const uint64x1_t vw = vshl_u64(vreinterpret_u64_u8(vld1_u8(w)), vshift);
        w += 8;
        const uint8x8x2_t vxy =
            vzip_u8(vreinterpret_u8_u64(vx), vreinterpret_u8_u64(vy));
        const uint8x8x2_t vzw =
            vzip_u8(vreinterpret_u8_u64(vz), vreinterpret_u8_u64(vw));
        const uint16x4x2_t vxyzw_lo = vzip_u16(
            vreinterpret_u16_u8(vxy.val[0]), vreinterpret_u16_u8(vzw.val[0]));
        const uint16x4x2_t vxyzw_hi = vzip_u16(
            vreinterpret_u16_u8(vxy.val[1]), vreinterpret_u16_u8(vzw.val[1]));

        uint32x2_t vxyzw0 = vreinterpret_u32_u16(vxyzw_lo.val[0]);
        uint32x2_t vxyzw1 = vreinterpret_u32_u16(vxyzw_lo.val[1]);
        uint32x2_t vxyzw2 = vreinterpret_u32_u16(vxyzw_hi.val[0]);
        uint32x2_t vxyzw3 = vreinterpret_u32_u16(vxyzw_hi.val[1]);

        if (k & 4) {
          vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 0);
          output = (void*)((uintptr_t)output + m);

          vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 1);
          output = (void*)((uintptr_t)output + m);

          vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw1, 0);
          output = (void*)((uintptr_t)output + m);

          vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw1, 1);
          output = (void*)((uintptr_t)output + m);

          vxyzw0 = vxyzw2;
          vxyzw1 = vxyzw3;
        }

        if (k & 2) {
          vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 0);
          output = (void*)((uintptr_t)output + m);

          vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 1);
          output = (void*)((uintptr_t)output + m);

          vxyzw0 = vxyzw1;
        }
        if (k & 1) {
          vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 0);
          output = (void*)((uintptr_t)output + m);
        }
      }
      output = (void*)((uintptr_t)output + output_increment);
      if (output > last_output) {
        output = last_output;
      }
    }
  } else {
    const uint8_t* i = input;
    uint8_t* o = output;
    size_t k = n;
    do {
      size_t l = m;
      const uint8_t* ii = i++;
      do {
        *o++ = *ii;
        ii += n;
      } while (--l != 0);
    } while (--k != 0);
  }
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip`):

- [`x4-neon.c_docs.md`](./x4-neon.c_docs.md)
- [`x2-sse2.c_docs.md`](./x2-sse2.c_docs.md)
- [`x4-sse2.c_docs.md`](./x4-sse2.c_docs.md)
- [`x2-neon.c_docs.md`](./x2-neon.c_docs.md)
- [`x3-sse2.c_docs.md`](./x3-sse2.c_docs.md)
- [`x3-neon.c_docs.md`](./x3-neon.c_docs.md)
- [`xm-sse2.c_docs.md`](./xm-sse2.c_docs.md)


## Cross-References

- **File Documentation**: `xm-neon.c_docs.md`
- **Keyword Index**: `xm-neon.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip`):

- [`x2-sse2.c_docs.md_docs.md`](./x2-sse2.c_docs.md_docs.md)
- [`x2-neon.c_docs.md_docs.md`](./x2-neon.c_docs.md_docs.md)
- [`x3-sse2.c_kw.md_docs.md`](./x3-sse2.c_kw.md_docs.md)
- [`xm-sse2.c_docs.md_docs.md`](./xm-sse2.c_docs.md_docs.md)
- [`x2-sse2.c_kw.md_docs.md`](./x2-sse2.c_kw.md_docs.md)
- [`x4-neon.c_kw.md_docs.md`](./x4-neon.c_kw.md_docs.md)
- [`x3-sse2.c_docs.md_docs.md`](./x3-sse2.c_docs.md_docs.md)
- [`x3-neon.c_docs.md_docs.md`](./x3-neon.c_docs.md_docs.md)
- [`x4-sse2.c_kw.md_docs.md`](./x4-sse2.c_kw.md_docs.md)
- [`x3-neon.c_kw.md_docs.md`](./x3-neon.c_kw.md_docs.md)


## Cross-References

- **File Documentation**: `xm-neon.c_docs.md_docs.md`
- **Keyword Index**: `xm-neon.c_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
