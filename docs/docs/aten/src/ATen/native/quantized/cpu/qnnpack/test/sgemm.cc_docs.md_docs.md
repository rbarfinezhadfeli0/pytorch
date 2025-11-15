# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/sgemm.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/sgemm.cc_docs.md`
- **Size**: 13,663 bytes (13.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/sgemm.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/sgemm.cc`
- **Size**: 11,000 bytes (10.74 KB)
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
#include <qnnpack/sgemm.h>

#include "gemm-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(SGEMM_5x8__NEON, k_eq_2) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(5).nr(8).np(8).kr(1).m(5).n(8).k(2).test(
      pytorch_sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8__NEON, k_eq_2_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .np(8)
      .kr(1)
      .m(5)
      .n(8)
      .k(2)
      .aStride(37)
      .test(pytorch_sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8__NEON, k_eq_2_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .np(8)
      .kr(1)
      .m(5)
      .n(8)
      .k(2)
      .cStride(17)
      .test(pytorch_sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8__NEON, k_eq_8_rmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_sgemm_ukernel_5x8__neon);
}

TEST(SGEMM_5x8__NEON, k_gt_2) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester().mr(5).nr(8).np(8).kr(1).m(5).n(8).k(k).test(
        pytorch_sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8__NEON, k_gt_2_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .np(8)
        .kr(1)
        .m(5)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8__NEON, k_gt_2_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .np(8)
        .kr(1)
        .m(5)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8__NEON, k_gt_2_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 3; k < 16; k++) {
    for (uint32_t m = 1; m <= 5; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_sgemm_ukernel_5x8__neon);
      }
    }
  }
}

TEST(SGEMM_5x8__NEON, k_div_2) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester().mr(5).nr(8).np(8).kr(1).m(5).n(8).k(k).test(
        pytorch_sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8__NEON, k_div_2_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .np(8)
        .kr(1)
        .m(5)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8__NEON, k_div_2_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .np(8)
        .kr(1)
        .m(5)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_sgemm_ukernel_5x8__neon);
  }
}

TEST(SGEMM_5x8__NEON, k_div_2_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 2; k < 32; k += 6) {
    for (uint32_t m = 1; m <= 5; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_sgemm_ukernel_5x8__neon);
      }
    }
  }
}

TEST(SGEMM_6x8__NEON, k_eq_2) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(2).test(
      pytorch_sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8__NEON, k_eq_2_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(2)
      .aStride(37)
      .test(pytorch_sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8__NEON, k_eq_2_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(2)
      .cStride(17)
      .test(pytorch_sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(8).qmin(128).test(
      pytorch_sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(8).qmax(128).test(
      pytorch_sgemm_ukernel_6x8__neon);
}

TEST(SGEMM_6x8__NEON, k_gt_2) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(k).test(
        pytorch_sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8__NEON, k_gt_2_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8__NEON, k_gt_2_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8__NEON, k_gt_2_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 3; k < 16; k++) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_sgemm_ukernel_6x8__neon);
      }
    }
  }
}

TEST(SGEMM_6x8__NEON, k_div_2) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(k).test(
        pytorch_sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8__NEON, k_div_2_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8__NEON, k_div_2_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_sgemm_ukernel_6x8__neon);
  }
}

TEST(SGEMM_6x8__NEON, k_div_2_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 2; k < 32; k += 6) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_sgemm_ukernel_6x8__neon);
      }
    }
  }
}
#endif

TEST(SGEMM_6x8__PSIMD, k_eq_2) {
  GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(2).test(
      pytorch_sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8__PSIMD, k_eq_2_strided_a) {
  GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(2)
      .aStride(37)
      .test(pytorch_sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8__PSIMD, k_eq_2_strided_c) {
  GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(2)
      .cStride(17)
      .test(pytorch_sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8__PSIMD, k_eq_8_qmin128) {
  GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(8).qmin(128).test(
      pytorch_sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8__PSIMD, k_eq_8_qmax128) {
  GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(8).qmax(128).test(
      pytorch_sgemm_ukernel_6x8__psimd);
}

TEST(SGEMM_6x8__PSIMD, k_gt_2) {
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(k).test(
        pytorch_sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8__PSIMD, k_gt_2_strided_a) {
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8__PSIMD, k_gt_2_strided_c) {
  for (size_t k = 3; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8__PSIMD, k_gt_2_subtile) {
  for (size_t k = 3; k < 16; k++) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_sgemm_ukernel_6x8__psimd);
      }
    }
  }
}

TEST(SGEMM_6x8__PSIMD, k_div_2) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(k).test(
        pytorch_sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8__PSIMD, k_div_2_strided_a) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8__PSIMD, k_div_2_strided_c) {
  for (size_t k = 2; k < 32; k += 2) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_sgemm_ukernel_6x8__psimd);
  }
}

TEST(SGEMM_6x8__PSIMD, k_div_2_subtile) {
  for (size_t k = 2; k < 32; k += 6) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_sgemm_ukernel_6x8__psimd);
      }
    }
  }
}

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
- `qnnpack/sgemm.h`
- `gemm-microkernel-tester.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/sgemm.cc
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

- **File Documentation**: `sgemm.cc_docs.md`
- **Keyword Index**: `sgemm.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

This is a test file. Run it with:

```bash
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/sgemm.cc_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/test`):

- [`leaky-relu.cc_kw.md_docs.md`](./leaky-relu.cc_kw.md_docs.md)
- [`sgemm.cc_kw.md_docs.md`](./sgemm.cc_kw.md_docs.md)
- [`softargmax-operator-tester.h_kw.md_docs.md`](./softargmax-operator-tester.h_kw.md_docs.md)
- [`maxpool-microkernel-tester.h_kw.md_docs.md`](./maxpool-microkernel-tester.h_kw.md_docs.md)
- [`rmax-microkernel-tester.h_kw.md_docs.md`](./rmax-microkernel-tester.h_kw.md_docs.md)
- [`add-operator-tester.h_kw.md_docs.md`](./add-operator-tester.h_kw.md_docs.md)
- [`tanh-operator-tester.h_docs.md_docs.md`](./tanh-operator-tester.h_docs.md_docs.md)
- [`channel-shuffle.cc_docs.md_docs.md`](./channel-shuffle.cc_docs.md_docs.md)
- [`q8vadd.cc_kw.md_docs.md`](./q8vadd.cc_kw.md_docs.md)
- [`global-average-pooling.cc_docs.md_docs.md`](./global-average-pooling.cc_docs.md_docs.md)


## Cross-References

- **File Documentation**: `sgemm.cc_docs.md_docs.md`
- **Keyword Index**: `sgemm.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
