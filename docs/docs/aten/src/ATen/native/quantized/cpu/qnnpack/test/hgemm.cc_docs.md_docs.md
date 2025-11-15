# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/hgemm.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/hgemm.cc_docs.md`
- **Size**: 7,267 bytes (7.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/hgemm.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/hgemm.cc`
- **Size**: 4,606 bytes (4.50 KB)
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

#include <qnnpack/hgemm.h>
#include <qnnpack/isa-checks.h>

#include "gemm-microkernel-tester.h"

#if CPUINFO_ARCH_ARM
TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).test(
      pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4_strided_a) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .aStride(37)
      .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4_strided_c) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .cStride(17)
      .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4_qmin128) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).qmin(128).test(
      pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4_qmax128) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).qmax(128).test(
      pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_gt_4) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 5; k < 8; k++) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_gt_4_strided_a) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 5; k < 8; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_gt_4_strided_c) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 5; k < 8; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_gt_4_subtile) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 5; k < 8; k++) {
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
      }
    }
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_div_4) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 8; k < 64; k += 4) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_div_4_strided_a) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 8; k < 64; k += 4) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_div_4_strided_c) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 8; k < 64; k += 4) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_div_4_subtile) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 8; k < 64; k += 12) {
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 8; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
      }
    }
  }
}
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

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
- `qnnpack/hgemm.h`
- `qnnpack/isa-checks.h`
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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/hgemm.cc
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

- **File Documentation**: `hgemm.cc_docs.md`
- **Keyword Index**: `hgemm.cc_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/hgemm.cc_docs.md
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

- **File Documentation**: `hgemm.cc_docs.md_docs.md`
- **Keyword Index**: `hgemm.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
