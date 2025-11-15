# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/global-average-pooling.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/global-average-pooling.cc_docs.md`
- **Size**: 23,821 bytes (23.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/global-average-pooling.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/global-average-pooling.cc`
- **Size**: 21,173 bytes (20.68 KB)
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

#include "global-average-pooling-operator-tester.h"

#include <qnnpack/params.h>

TEST(GLOBAL_AVERAGE_POOLING_OP, zero_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  GlobalAveragePoolingOperatorTester()
      .batchSize(0)
      .width(1)
      .channels(8)
      .testQ8();
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_many_channels_small_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_output_min) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMin(128)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_output_max) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMax(128)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_many_channels_large_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_output_min) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMin(128)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_output_max) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMax(128)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_few_channels_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_min) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMin(128)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_max) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMax(128)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_many_channels_small_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    small_batch_many_channels_small_width_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    small_batch_many_channels_small_width_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_many_channels_large_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    small_batch_many_channels_large_width_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    small_batch_many_channels_large_width_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_few_channels) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_few_channels_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_few_channels_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
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
- `global-average-pooling-operator-tester.h`
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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/global-average-pooling.cc
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
- [`channel-shuffle-operator-tester.h_docs.md`](./channel-shuffle-operator-tester.h_docs.md)


## Cross-References

- **File Documentation**: `global-average-pooling.cc_docs.md`
- **Keyword Index**: `global-average-pooling.cc_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/global-average-pooling.cc_docs.md
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


## Cross-References

- **File Documentation**: `global-average-pooling.cc_docs.md_docs.md`
- **Keyword Index**: `global-average-pooling.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
