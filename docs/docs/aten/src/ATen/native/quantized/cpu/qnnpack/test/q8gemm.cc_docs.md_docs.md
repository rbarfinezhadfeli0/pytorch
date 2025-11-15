# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm.cc_docs.md`
- **Size**: 52,669 bytes (51.43 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm.cc`
- **Size**: 81,961 bytes (80.04 KB)
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
#include <qnnpack/q8gemm.h>

#include "gemm-microkernel-tester.h"

#if CPUINFO_ARCH_ARM
TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
      pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
      }
    }
  }
}

//
// Dynamic Quantization
//

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
      pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_DQ_4x8__AARCH32_NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
        pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
        pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_ARM64
TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).test(
      pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_strided_a) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_strided_c) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_qmin128) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_qmax128) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_azp0) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_bzp0) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_nozp) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_strided_a) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_strided_c) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_azp0) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_bzp0) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_nozp) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_subtile) {
  for (size_t k = 9; k < 16; k++) {
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
            .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
      }
    }
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_div_8) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_div_8_strided_a) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_div_8_strided_c) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_div_8_subtile) {
  for (size_t k = 16; k < 128; k += 24) {
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
            .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
      }
    }
  }
}

//
// Dynamic Quantization
//

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_eq_8) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).test(
      pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_eq_8_strided_a) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_eq_8_strided_c) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_eq_8_qmin128) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_eq_8_qmax128) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_eq_8_azp0) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_eq_8_bzp0) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_eq_8_nozp) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_gt_8) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_gt_8_strided_a) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_gt_8_strided_c) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_gt_8_azp0) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_gt_8_bzp0) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_gt_8_nozp) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_gt_8_subtile) {
  for (size_t k = 9; k < 16; k++) {
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
            .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
      }
    }
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_div_8) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_div_8_strided_a) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_div_8_strided_c) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_DQ_8x8__AARCH64_NEON, k_div_8_subtile) {
  for (size_t k = 16; k < 128; k += 24) {
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
            .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8GEMM_4x8__NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
      pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8__NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x8__neon);
      }
    }
  }
}

//
// Dynamic Quantization
//

TEST(Q8GEMM_DQ_4x8__NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
      pytorch_q8gemm_dq_ukernel_4x8__neon);
}

TEST(Q8GEMM_DQ_4x8__NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
}

TEST(Q8GEMM_DQ_4x8__NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
}

TEST(Q8GEMM_DQ_4x8__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_dq_ukernel_4x8__neon);
}

TEST(Q8GEMM_DQ_4x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_dq_ukernel_4x8__neon);
}

TEST(Q8GEMM_DQ_4x8__NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
}

TEST(Q8GEMM_DQ_4x8__NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
}

TEST(Q8GEMM_DQ_4x8__NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
}

TEST(Q8GEMM_DQ_4x8__NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_DQ_4x8__NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_8x8__NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).test(
      pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
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
            .test(pytorch_q8gemm_ukernel_8x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_8x8__NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
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
            .test(pytorch_q8gemm_ukernel_8x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_6x4__NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).test(
      pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(k).test(
        pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(4)
            .np(4)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_6x4__neon);
      }
    }
  }
}

TEST(Q8GEMM_6x4__NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(k).test(
        pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm.cc_docs.md
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

- **File Documentation**: `q8gemm.cc_docs.md_docs.md`
- **Keyword Index**: `q8gemm.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
