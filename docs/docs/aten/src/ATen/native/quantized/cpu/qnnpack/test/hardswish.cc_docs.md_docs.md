# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/hardswish.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/hardswish.cc_docs.md`
- **Size**: 9,258 bytes (9.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/hardswish.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/hardswish.cc`
- **Size**: 6,613 bytes (6.46 KB)
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

#include "hardswish-operator-tester.h"

#include <qnnpack/params.h>

TEST(HARDSWISH_OP, zero_batch) {
  HardswishOperatorTester().batchSize(0).channels(8).iterations(1).testQ8();
}

TEST(HARDSWISH_OP, unit_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(1)
        .channels(channels)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, unit_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 10.0f) {
      HardswishOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputScale(inputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(HARDSWISH_OP, unit_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      HardswishOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputZeroPoint(uint8_t(inputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(HARDSWISH_OP, unit_batch_with_output_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float outputScale = 1.0e-2f; outputScale < 1.0e+2f;
         outputScale *= 10.0f) {
      HardswishOperatorTester()
          .batchSize(1)
          .channels(channels)
          .outputScale(outputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(HARDSWISH_OP, unit_batch_with_output_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
         outputZeroPoint += 51) {
      HardswishOperatorTester()
          .batchSize(1)
          .channels(channels)
          .outputZeroPoint(uint8_t(outputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(HARDSWISH_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(3)
        .channels(channels)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(3)
        .channels(channels)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, small_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, small_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, small_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 10.0f) {
      HardswishOperatorTester()
          .batchSize(3)
          .channels(channels)
          .inputScale(inputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(HARDSWISH_OP, small_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      HardswishOperatorTester()
          .batchSize(3)
          .channels(channels)
          .inputZeroPoint(uint8_t(inputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(HARDSWISH_OP, strided_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, strided_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, strided_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    HardswishOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(HARDSWISH_OP, strided_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 10.0f) {
      HardswishOperatorTester()
          .batchSize(3)
          .channels(channels)
          .inputStride(129)
          .outputStride(117)
          .inputScale(inputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(HARDSWISH_OP, strided_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      HardswishOperatorTester()
          .batchSize(3)
          .channels(channels)
          .inputStride(129)
          .outputStride(117)
          .inputZeroPoint(uint8_t(inputZeroPoint))
          .iterations(1)
          .testQ8();
    }
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
- `hardswish-operator-tester.h`
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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/hardswish.cc
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

- **File Documentation**: `hardswish.cc_docs.md`
- **Keyword Index**: `hardswish.cc_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/hardswish.cc_docs.md
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

- **File Documentation**: `hardswish.cc_docs.md_docs.md`
- **Keyword Index**: `hardswish.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
