# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8vadd.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8vadd.cc`
- **Size**: 7,838 bytes (7.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <qnnpack/isa-checks.h>
#include <qnnpack/q8vadd.h>

#include "vadd-microkernel-tester.h"

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8VADD__SSE2, n_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  VAddMicrokernelTester().n(8).test(pytorch_q8vadd_ukernel__sse2);
}

TEST(Q8VADD__SSE2, n_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, n_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, n_lt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, inplace_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).inplaceA(true).test(
        pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, inplace_b) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).inplaceB(true).test(
        pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, inplace_a_and_b) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplaceA(true)
        .inplaceB(true)
        .test(pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, a_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (float aScale = 1.0e-2; aScale < 1.0e+2; aScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).aScale(aScale).test(
          pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, b_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (float bScale = 1.0e-2; bScale < 1.0e+2; bScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).bScale(bScale).test(
          pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (float yScale = 1.0e-2; yScale < 1.0e+2; yScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).yScale(yScale).test(
          pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, a_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .aZeroPoint(uint8_t(aZeroPoint))
          .test(pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, b_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .bZeroPoint(uint8_t(bZeroPoint))
          .test(pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .yZeroPoint(uint8_t(yZeroPoint))
          .test(pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).qmin(128).test(
        pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).qmax(128).test(
        pytorch_q8vadd_ukernel__sse2);
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8VADD__NEON, n_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  VAddMicrokernelTester().n(8).test(pytorch_q8vadd_ukernel__neon);
}

TEST(Q8VADD__NEON, n_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, n_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, n_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, inplace_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).inplaceA(true).test(
        pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, inplace_b) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).inplaceB(true).test(
        pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, inplace_a_and_b) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplaceA(true)
        .inplaceB(true)
        .test(pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, a_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (float aScale = 1.0e-2; aScale < 1.0e+2; aScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).aScale(aScale).test(
          pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, b_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (float bScale = 1.0e-2; bScale < 1.0e+2; bScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).bScale(bScale).test(
          pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (float yScale = 1.0e-2; yScale < 1.0e+2; yScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).yScale(yScale).test(
          pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, a_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .aZeroPoint(uint8_t(aZeroPoint))
          .test(pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, b_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .bZeroPoint(uint8_t(bZeroPoint))
          .test(pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .yZeroPoint(uint8_t(yZeroPoint))
          .test(pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).qmin(128).test(
        pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).qmax(128).test(
        pytorch_q8vadd_ukernel__neon);
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cpuinfo.h`
- `gtest/gtest.h`
- `qnnpack/isa-checks.h`
- `qnnpack/q8vadd.h`
- `vadd-microkernel-tester.h`


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

This is a test file. Run it with:

```bash
python aten/src/ATen/native/quantized/cpu/qnnpack/test/q8vadd.cc
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/test`):

- [`avgpool-microkernel-tester.h_docs.md`](./avgpool-microkernel-tester.h_docs.md)
- [`tanh.cc_docs.md`](./tanh.cc_docs.md)
- [`average-pooling-operator-tester.h_docs.md`](./average-pooling-operator-tester.h_docs.md)
- [`u8lut32norm.cc_docs.md`](./u8lut32norm.cc_docs.md)
- [`lut-norm-microkernel-tester.h_docs.md`](./lut-norm-microkernel-tester.h_docs.md)
- [`softargmax.cc_docs.md`](./softargmax.cc_docs.md)
- [`hardsigmoid-operator-tester.h_docs.md`](./hardsigmoid-operator-tester.h_docs.md)
- [`q8avgpool.cc_docs.md`](./q8avgpool.cc_docs.md)
- [`global-average-pooling.cc_docs.md`](./global-average-pooling.cc_docs.md)
- [`channel-shuffle-operator-tester.h_docs.md`](./channel-shuffle-operator-tester.h_docs.md)


## Cross-References

- **File Documentation**: `q8vadd.cc_docs.md`
- **Keyword Index**: `q8vadd.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
