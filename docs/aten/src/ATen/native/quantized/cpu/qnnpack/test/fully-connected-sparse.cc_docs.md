# Documentation: fully-connected-sparse.cc

## File Metadata
- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse.cc`
- **Size**: 4452 bytes
- **Lines**: 144
- **Extension**: .cc
- **Type**: Regular file

## Original Source

```cc
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fully-connected-sparse-operator-tester.h"

#define SPARSE_OP_TEST(ROW_BS, COL_BS) \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    integration_test_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(4) \
      .inputChannels(4) \
      .outputChannels(4) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    zero_batch_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(0) \
      .inputChannels(2) \
      .outputChannels(2) \
      .iterations(1) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    unit_batch_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(1) \
      .inputChannels(23) \
      .outputChannels(19) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    unit_batch_with_qmin_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(1) \
      .inputChannels(23) \
      .outputChannels(19) \
      .qmin(128) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    unit_batch_with_qmax_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(1) \
      .inputChannels(23) \
      .outputChannels(19) \
      .qmax(128) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    unit_batch_with_input_stride_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(1) \
      .inputChannels(23) \
      .inputStride(28) \
      .outputChannels(19) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    unit_batch_with_output_stride_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(1) \
      .inputChannels(23) \
      .outputChannels(19) \
      .outputStride(29) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    small_batch_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(12) \
      .inputChannels(23) \
      .outputChannels(19) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    small_batch_with_qmin_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(12) \
      .inputChannels(23) \
      .outputChannels(19) \
      .qmin(128) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
} \
 \
TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, \
    small_batch_with_qmax_dynamic_prepacked) { \
  FullyConnectedSparseOperatorTester() \
      .batchSize(13) \
      .inputChannels(23) \
      .outputChannels(19) \
      .qmax(128) \
      .iterations(3) \
      .rowBlockSize(ROW_BS) \
      .colBlockSize(COL_BS) \
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); \
}

SPARSE_OP_TEST(1, 4)
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
SPARSE_OP_TEST(8, 1)
#endif

```

## High-Level Overview

This file is part of the PyTorch repository. It is a source or configuration file.

## Detailed Walkthrough


## Key Components

The file contains 337 words across 144 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4452 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
