# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/clamp.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/clamp.cc`
- **Size**: 2,532 bytes (2.47 KB)
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

#include "clamp-operator-tester.h"

TEST(CLAMP_OP, zero_batch) {
  ClampOperatorTester().batchSize(0).channels(2).iterations(1).testU8();
}

TEST(CLAMP_OP, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
        .batchSize(1)
        .channels(channels)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampOperatorTester()
          .batchSize(1)
          .channels(channels)
          .qmin(qmin)
          .iterations(3)
          .testU8();
    }
  }
}

TEST(CLAMP_OP, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampOperatorTester()
          .batchSize(1)
          .channels(channels)
          .qmax(qmax)
          .iterations(3)
          .testU8();
    }
  }
}

TEST(CLAMP_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .outputStride(117)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, qmin_and_qmax_equal_uint8_max) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmin(255)
        .qmax(255)
        .iterations(3)
        .testU8();
  }
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

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
- `clamp-operator-tester.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/clamp.cc
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

- **File Documentation**: `clamp.cc_docs.md`
- **Keyword Index**: `clamp.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
