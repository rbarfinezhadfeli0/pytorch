# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8dwconv.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8dwconv.cc_docs.md`
- **Size**: 52,683 bytes (51.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8dwconv.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8dwconv.cc`
- **Size**: 67,373 bytes (65.79 KB)
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
#include <qnnpack/q8dwconv.h>

#include "dwconv-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_gt_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_gt_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(3)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8_with_qmin_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8_with_qmax_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_eq_8_with_input_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    multi_output_channels_eq_8_with_subsampling_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    multi_output_channels_eq_8_with_input_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    multi_output_channels_eq_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_div_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_div_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    multi_output_channels_div_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8_with_qmin_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8_with_qmax_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_gt_8_with_input_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_gt_8_with_kernel_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_gt_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    multi_output_channels_gt_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_eq_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    multi_output_channels_eq_8_with_subsampling_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    multi_output_channels_eq_8_with_input_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    multi_output_channels_eq_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_eq_8_with_qmin_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_eq_8_with_qmax_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_eq_8_with_input_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(3)
      .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_div_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_div_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
  }
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    multi_output_channels_div_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_gt_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
  }
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_gt_8_with_qmin_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
  }
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_gt_8_with_qmax_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_gt_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
  }
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    multi_output_channels_gt_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_ARM
TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_gt_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_gt_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8_with_qmin_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8_with_qmax_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_eq_8_with_input_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_eq_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_subsampling_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_input_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_div_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_div_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_div_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8_with_qmin_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8_with_qmax_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_gt_8_with_input_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_gt_8_with_kernel_zero_point_only_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_gt_8_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_gt_8_with_output_stride_per_channel) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
  }
}
#endif /* CPUINFO_ARCH_ARM */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(
    Q8DWCONV_UP8x9__SSE2,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(
    Q8DWCONV_UP8x9__SSE2,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(
    Q8DWCONV_UP8x9__SSE2,
    single_output_channels_gt_8_with_input_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(
    Q8DWCONV_UP8x9__SSE2,
    single_output_channels_gt_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(
    Q8DWCONV_MP8x25__SSE2,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(
    Q8DWCONV_MP8x25__SSE2,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_uk
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8dwconv.cc_docs.md
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

- **File Documentation**: `q8dwconv.cc_docs.md_docs.md`
- **Keyword Index**: `q8dwconv.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
