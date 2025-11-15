# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gavgpool.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gavgpool.cc_docs.md`
- **Size**: 35,333 bytes (34.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gavgpool.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gavgpool.cc`
- **Size**: 32,636 bytes (31.87 KB)
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
#include <qnnpack/q8gavgpool.h>

#include "gavgpool-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester().m(7).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(m).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester().m(7).n(8).xStride(11).test(
      pytorch_q8gavgpool_ukernel_up8x7__neon);
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(7).n(8).xScale(xScale).test(
        pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .xZeroPoint(xZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(7).n(8).yScale(yScale).test(
        pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .yZeroPoint(yZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMax(128)
      .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMin(128)
      .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_div_8_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_div_8_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
      GAvgPoolMicrokernelTester().m(7).n(n).xScale(xScale).test(
          pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
      GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .xZeroPoint(xZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
      GAvgPoolMicrokernelTester().m(7).n(n).yScale(yScale).test(
          pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .yZeroPoint(yZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMax(128)
        .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMin(128)
        .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester().m(14).n(8).nr(8).test(
      pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xStride(11).test(
      pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xScale(xScale).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .nr(8)
        .xZeroPoint(xZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(14).n(8).nr(8).yScale(yScale).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .nr(8)
        .yZeroPoint(yZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .nr(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMax(128)
      .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .nr(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMin(128)
      .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_few_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).xStride(11).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_multipass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_div_8_2pass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_div_8_2pass_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_div_8_multipass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_div_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).xStride(131).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester().m(14).n(n).nr(8).xScale(xScale).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .nr(8)
          .xZeroPoint(xZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester().m(14).n(n).nr(8).yScale(yScale).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .nr(8)
          .yZeroPoint(yZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .nr(8)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMax(128)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .nr(8)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMin(128)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_multipass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(23).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_small_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 8; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8xm__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_large_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 8; m < 16; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8xm__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
        GAvgPoolMicrokernelTester().m(m).n(n).xScale(xScale).test(
            pytorch_q8gavgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
        GAvgPoolMicrokernelTester()
            .m(m)
            .n(n)
            .xZeroPoint(xZeroPoint)
            .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
        GAvgPoolMicrokernelTester().m(m).n(n).yScale(yScale).test(
            pytorch_q8gavgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
        GAvgPoolMicrokernelTester()
            .m(m)
            .n(n)
            .yZeroPoint(yZeroPoint)
            .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMax(128)
          .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMin(128)
          .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
    }
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester().m(7).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(m).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester().m(7).n(8).xStride(11).test(
      pytorch_q8gavgpool_ukernel_up8x7__sse2);
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(7).n(8).xScale(xScale).test(
        pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .xZeroPoint(xZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(7).n(8).yScale(yScale).test(
        pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .yZeroPoint(yZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMax(128)
      .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMin(128)
      .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_div_8_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_div_8_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
      GAvgPoolMicrokernelTester().m(7).n(n).xScale(xScale).test(
          pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
      GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .xZeroPoint(xZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
      GAvgPoolMicrokernelTester().m(7).n(n).yScale(yScale).test(
          pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .yZeroPoint(yZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMax(128)
        .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMin(128)
        .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester().m(14).n(8).nr(8).test(
      pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xStride(11).test(
      pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xScale(xScale).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .nr(8)
        .xZeroPoint(xZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(14).n(8).nr(8).yScale(yScale).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .nr(8)
        .yZeroPoint(yZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .nr(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMax(128)
      .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .nr(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMin(128)
      .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_few_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).xStride(11).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_multipass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_div_8_2pass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_div_8_2pass_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_div_8_multipass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_div_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).xStride(131).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester().m(14).n(n).nr(8).xScale(xScale).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .nr(8)
          .xZeroPoint(xZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester().m(14).n(n).nr(8).yScale(yScale).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .nr(8)
          .yZeroPoint(yZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .nr(8)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMax(128)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .nr(8)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMin(128)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_multipass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(23).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_small_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 8; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8xm__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_large_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 8; m < 16; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8xm__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
        GAvgPoolMicrokernelTester().m(m).n(n).xScale(xScale).test(
            pytorch_q8gavgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
        GAvgPoolMicrokernelTester()
            .m(m)
            .n(n)
            .xZeroPoint(xZeroPoint)
            .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
        GAvgPoolMicrokernelTester().m(m).n(n).yScale(yScale).test(
            pytorch_q8gavgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
        GAvgPoolMicrokernelTester()
            .m(m)
            .n(n)
            .yZeroPoint(yZeroPoint)
            .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMax(128)
          .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMin(128)
          .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
    }
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

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
- `qnnpack/q8gavgpool.h`
- `gavgpool-microkernel-tester.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gavgpool.cc
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

- **File Documentation**: `q8gavgpool.cc_docs.md`
- **Keyword Index**: `q8gavgpool.cc_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gavgpool.cc_docs.md
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

- **File Documentation**: `q8gavgpool.cc_docs.md_docs.md`
- **Keyword Index**: `q8gavgpool.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
