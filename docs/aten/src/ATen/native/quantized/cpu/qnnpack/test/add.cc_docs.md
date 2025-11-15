# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/add.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/add.cc`
- **Size**: 10,084 bytes (9.85 KB)
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

#include "add-operator-tester.h"

TEST(ADD_OP, zero_batch) {
  AddOperatorTester().batchSize(0).channels(2).iterations(1).testQ8();
}

TEST(ADD_OP, unit_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester().batchSize(1).channels(channels).iterations(3).testQ8();
  }
}

TEST(ADD_OP, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, unit_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .aScale(aScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .bScale(bScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .yScale(yScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .aZeroPoint(uint8_t(aZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .bZeroPoint(uint8_t(bZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .yZeroPoint(uint8_t(yZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester().batchSize(3).channels(channels).iterations(3).testQ8();
  }
}

TEST(ADD_OP, small_batch_with_a_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .aStride(129)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_b_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .bStride(123)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_y_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .yStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aScale(aScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .bScale(bScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .yScale(yScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aZeroPoint(uint8_t(aZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .bZeroPoint(uint8_t(bZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .yZeroPoint(uint8_t(yZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .aStride(129)
        .bStride(123)
        .yStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, strided_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .aStride(129)
        .bStride(123)
        .yStride(117)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, strided_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .aStride(129)
        .bStride(123)
        .yStride(117)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, strided_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .aScale(aScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .bScale(bScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .yScale(yScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .aZeroPoint(uint8_t(aZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .bZeroPoint(uint8_t(bZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .yZeroPoint(uint8_t(yZeroPoint))
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
- `add-operator-tester.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/add.cc
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

- **File Documentation**: `add.cc_docs.md`
- **Keyword Index**: `add.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
