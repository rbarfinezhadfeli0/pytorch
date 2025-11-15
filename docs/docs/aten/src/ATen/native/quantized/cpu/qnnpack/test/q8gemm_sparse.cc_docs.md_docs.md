# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm_sparse.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm_sparse.cc_docs.md`
- **Size**: 52,724 bytes (51.49 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm_sparse.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm_sparse.cc`
- **Size**: 51,912 bytes (50.70 KB)
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
#include <qnnpack/q8gemm_sparse.h>

#include "gemm-block-sparse-microkernel-tester.h"

#define TEST_PACKED_ROW_BLOCK_SIZEXCOL_BLOCK_SIZE_SPARSE_OP(MR, \
    NR, row_block_size, col_block_size, \
    prepacking_kernel, compute_kernel_w32, compute_kernel_w16, compute_kernel_w8) \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_4) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
\
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_4_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aStride(37); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_4_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .cStride(17); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_4_qmin128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .qmin(128); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_4_qmax128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .qmax(128); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_4_azp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_4_bzp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .bZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_4_nozp) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .bZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_8) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_8_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aStride(37); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_8_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .cStride(17); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_8_qmin128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .qmin(128); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_8_qmax128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .qmax(128); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_8_azp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_8_bzp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .bZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_lt_8_nozp) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .bZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_eq_8) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_eq_8_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aStride(37); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_eq_8_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .cStride(17); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_eq_8_qmin128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .qmin(128); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_eq_8_qmax128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .qmax(128); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_eq_8_azp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_eq_8_bzp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .bZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_eq_8_nozp) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .bZeroPoint(0); \
  tester.test_packed<uint32_t>( \
      prepacking_kernel, \
      compute_kernel_w32); \
  tester.test_packed<uint16_t>( \
      prepacking_kernel, \
      compute_kernel_w16); \
  tester.test_packed<uint8_t>( \
      prepacking_kernel, \
      compute_kernel_w8); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_gt_8) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_gt_8_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .aStride(37); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_gt_8_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .cStride(17); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_gt_8_azp0) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .aZeroPoint(0); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_gt_8_bzp0) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .bZeroPoint(0); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_gt_8_nozp) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .aZeroPoint(0) \
        .bZeroPoint(0); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_gt_8_subtile) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    for (uint32_t m = 1; m <= MR; m++) { \
      for (uint32_t n = 1; n <= NR; n++) { \
        GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
            .mr(MR) \
            .nr(NR) \
            .m(m) \
            .n(n) \
            .k(k) \
            .rowBlockSize(row_block_size) \
            .colBlockSize(col_block_size) \
            .iterations(3); \
        tester.test_packed<uint32_t>( \
            prepacking_kernel, \
            compute_kernel_w32); \
        tester.test_packed<uint16_t>( \
            prepacking_kernel, \
            compute_kernel_w16); \
        tester.test_packed<uint8_t>( \
            prepacking_kernel, \
            compute_kernel_w8); \
      } \
    } \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_div_8) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 16; k < 128; k += 8) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_div_8_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 16; k < 128; k += 8) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .aStride(171); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_div_8_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 16; k < 128; k += 8) { \
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .cStride(17); \
    tester.test_packed<uint32_t>( \
        prepacking_kernel, \
        compute_kernel_w32); \
    tester.test_packed<uint16_t>( \
        prepacking_kernel, \
        compute_kernel_w16); \
    tester.test_packed<uint8_t>( \
        prepacking_kernel, \
        compute_kernel_w8); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size ## __AARCH32_NEON, packedA_k_div_8_subtile) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 16; k < 128; k += 24) { \
    for (uint32_t m = 1; m <= MR; m++) { \
      for (uint32_t n = 1; n <= NR; n++) { \
        GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester() \
            .mr(MR) \
            .nr(NR) \
            .m(m) \
            .n(n) \
            .k(k) \
            .rowBlockSize(row_block_size) \
            .colBlockSize(col_block_size) \
            .iterations(3); \
        tester.test_packed<uint32_t>( \
            prepacking_kernel, \
            compute_kernel_w32); \
        tester.test_packed<uint16_t>( \
            prepacking_kernel, \
            compute_kernel_w16); \
        tester.test_packed<uint8_t>( \
            prepacking_kernel, \
            compute_kernel_w8); \
      } \
    } \
  } \
}

#define TEST_PACKED_1x4_SPARSE_OP(MR, NR, prepacking_kernel, compute_kernel_w32, compute_kernel_w16, compute_kernel_w8) \
  TEST_PACKED_ROW_BLOCK_SIZEXCOL_BLOCK_SIZE_SPARSE_OP(MR, \
      NR, 1, 4, prepacking_kernel, compute_kernel_w32, compute_kernel_w16, compute_kernel_w8)
#define TEST_PACKED_8x1_SPARSE_OP(MR, NR, prepacking_kernel, compute_kernel_w32, compute_kernel_w16, compute_kernel_w8) \
  TEST_PACKED_ROW_BLOCK_SIZEXCOL_BLOCK_SIZE_SPARSE_OP(MR, \
      NR, 8, 1, prepacking_kernel, compute_kernel_w32, compute_kernel_w16, compute_kernel_w8)

#if CPUINFO_ARCH_ARM
TEST_PACKED_1x4_SPARSE_OP(
    4,
    8,
    pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w32__aarch32_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w16__aarch32_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w8__aarch32_neon)
TEST_PACKED_8x1_SPARSE_OP(
    4,
    8,
    pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w32__aarch32_neon,
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w16__aarch32_neon,
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w8__aarch32_neon)

#endif

#if CPUINFO_ARCH_ARM64

TEST_PACKED_1x4_SPARSE_OP(
    8,
    8,
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w32__aarch64_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w16__aarch64_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w8__aarch64_neon)
TEST_PACKED_8x1_SPARSE_OP(
    8,
    8,
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon,
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w32__aarch64_neon,
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w16__aarch64_neon,
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w8__aarch64_neon)

#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aStride(37);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .cStride(17);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .qmin(128);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .qmax(128);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .bZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .bZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aStride(37);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .cStride(17);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .qmin(128);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .qmax(128);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .bZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .bZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aStride(37);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .cStride(17);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .qmin(128);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .qmax(128);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .bZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0);
  tester.test_packed<uint32_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
  tester.test_packed<uint16_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
  tester.test_packed<uint8_t>(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k);
    tester.test_packed<uint32_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
    tester.test_packed<uint16_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
    tester.test_packed<uint8_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(37);
    tester.test_packed<uint32_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
    tester.test_packed<uint16_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
    tester.test_packed<uint8_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17);
    tester.test_packed<uint32_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
    tester.test_packed<uint16_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
    tester.test_packed<uint8_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0);
    tester.test_packed<uint32_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
    tester.test_packed<uint16_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
    tester.test_packed<uint8_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .bZeroPoint(0);
    tester.test_packed<uint32_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
    tester.test_packed<uint16_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
    tester.test_packed<uint8_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0);
    tester.test_packed<uint32_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
    tester.test_packed<uint16_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
    tester.test_packed<uint8_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k);
    tester.test_packed<uint32_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2);
    tester.test_packed<uint16_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2);
    tester.test_packed<uint8_t>(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_div_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester tester = GemmBlockSparseMicrokernelTester()
        .mr(8)
 
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm_sparse.cc_docs.md
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

- **File Documentation**: `q8gemm_sparse.cc_docs.md_docs.md`
- **Keyword Index**: `q8gemm_sparse.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
