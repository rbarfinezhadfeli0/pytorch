# Documentation: sigmoid.cc

## File Metadata
- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/sigmoid.cc`
- **Size**: 5739 bytes
- **Lines**: 229
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

#include "sigmoid-operator-tester.h"

#include <qnnpack/params.h>

TEST(SIGMOID_OP, zero_batch) {
  SigmoidOperatorTester().batchSize(0).channels(8).iterations(1).testQ8();
}

TEST(SIGMOID_OP, unit_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(1)
        .channels(channels)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, unit_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 10.0f) {
      SigmoidOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputScale(inputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(SIGMOID_OP, unit_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      SigmoidOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputZeroPoint(uint8_t(inputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(SIGMOID_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(3)
        .channels(channels)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(3)
        .channels(channels)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, small_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, small_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, small_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 10.0f) {
      SigmoidOperatorTester()
          .batchSize(3)
          .channels(channels)
          .inputScale(inputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(SIGMOID_OP, small_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      SigmoidOperatorTester()
          .batchSize(3)
          .channels(channels)
          .inputZeroPoint(uint8_t(inputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(SIGMOID_OP, strided_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, strided_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, strided_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    SigmoidOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(SIGMOID_OP, strided_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 10.0f) {
      SigmoidOperatorTester()
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

TEST(SIGMOID_OP, strided_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      SigmoidOperatorTester()
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

This file is part of the PyTorch repository. It is a source or configuration file.

## Detailed Walkthrough


## Key Components

The file contains 526 words across 229 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5739 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
