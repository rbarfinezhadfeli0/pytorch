# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse.cc_docs.md`
- **Size**: 7,154 bytes (6.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse.cc`
- **Size**: 4,452 bytes (4.35 KB)
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

- `gtest/gtest.h`
- `fully-connected-sparse-operator-tester.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse.cc
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

- **File Documentation**: `fully-connected-sparse.cc_docs.md`
- **Keyword Index**: `fully-connected-sparse.cc_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse.cc_docs.md
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

- **File Documentation**: `fully-connected-sparse.cc_docs.md_docs.md`
- **Keyword Index**: `fully-connected-sparse.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
