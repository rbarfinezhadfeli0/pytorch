# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/softargmax.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/softargmax.cc`
- **Size**: 3,443 bytes (3.36 KB)
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

#include "softargmax-operator-tester.h"

#include <qnnpack/params.h>

TEST(SOFTARGMAX_OP, zero_batch) {
  SoftArgMaxOperatorTester().batchSize(0).channels(1).iterations(1).testQ8();
}

TEST(SOFTARGMAX_OP, single_class) {
  SoftArgMaxOperatorTester().batchSize(1).channels(1).iterations(100).testQ8();
}

TEST(SOFTARGMAX_OP, two_classes) {
  SoftArgMaxOperatorTester().batchSize(1).channels(2).iterations(100).testQ8();
}

TEST(SOFTARGMAX_OP, many_classes) {
  for (size_t channels = 3; channels < 100; channels++) {
    SoftArgMaxOperatorTester()
        .batchSize(1)
        .channels(channels)
        .iterations(1)
        .testQ8();
  }
}

TEST(SOFTARGMAX_OP, cifar_classes) {
  /* CIFAR-10 */
  SoftArgMaxOperatorTester().batchSize(1).channels(10).iterations(15).testQ8();
  /* CIFAR-100 */
  SoftArgMaxOperatorTester().batchSize(1).channels(100).iterations(15).testQ8();
}

TEST(SOFTARGMAX_OP, imagenet_classes) {
  /* ImageNet-1K */
  SoftArgMaxOperatorTester()
      .batchSize(1)
      .channels(1000)
      .iterations(10)
      .testQ8();
  /* ImageNet-1K+1 */
  SoftArgMaxOperatorTester()
      .batchSize(1)
      .channels(1001)
      .iterations(10)
      .testQ8();
  /* ImageNet-22K */
  SoftArgMaxOperatorTester()
      .batchSize(1)
      .channels(21841)
      .iterations(10)
      .testQ8();
}

TEST(SOFTARGMAX_OP, many_channels_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 3.14159265f) {
      SoftArgMaxOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputScale(inputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(SOFTARGMAX_OP, many_channels_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      SoftArgMaxOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputZeroPoint(uint8_t(inputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(SOFTARGMAX_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftArgMaxOperatorTester()
        .batchSize(3)
        .channels(channels)
        .iterations(3)
        .testQ8();
  }
}

TEST(SOFTARGMAX_OP, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftArgMaxOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .iterations(3)
        .testQ8();
  }
}

TEST(SOFTARGMAX_OP, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftArgMaxOperatorTester()
        .batchSize(3)
        .channels(channels)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(SOFTARGMAX_OP, strided_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftArgMaxOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .iterations(3)
        .testQ8();
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
- `softargmax-operator-tester.h`
- `qnnpack/params.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/softargmax.cc
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
- [`hardsigmoid-operator-tester.h_docs.md`](./hardsigmoid-operator-tester.h_docs.md)
- [`q8avgpool.cc_docs.md`](./q8avgpool.cc_docs.md)
- [`global-average-pooling.cc_docs.md`](./global-average-pooling.cc_docs.md)
- [`channel-shuffle-operator-tester.h_docs.md`](./channel-shuffle-operator-tester.h_docs.md)


## Cross-References

- **File Documentation**: `softargmax.cc_docs.md`
- **Keyword Index**: `softargmax.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
