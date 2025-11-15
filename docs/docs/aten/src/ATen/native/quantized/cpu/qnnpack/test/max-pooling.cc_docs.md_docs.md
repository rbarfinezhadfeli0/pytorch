# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/max-pooling.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/max-pooling.cc_docs.md`
- **Size**: 42,329 bytes (41.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/max-pooling.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/max-pooling.cc`
- **Size**: 39,670 bytes (38.74 KB)
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

#include "max-pooling-operator-tester.h"

#include <qnnpack/params.h>

TEST(MAX_POOLING_OP, zero_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  MaxPoolingOperatorTester()
      .batchSize(0)
      .inputHeight(2)
      .inputWidth(6)
      .poolingHeight(1)
      .poolingWidth(8)
      .channels(8)
      .testU8();
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_1xM_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_1xM_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 3; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      for (size_t paddingWidth = 0; paddingWidth <= 1; paddingWidth++) {
        for (size_t paddingRight = 0; paddingRight <= 1; paddingRight++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .paddingWidth(paddingWidth)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_1xM_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 4)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .strideWidth(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_1xM_pool_with_dilation) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(2 * poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .dilationWidth(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_Mx1_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_Mx1_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      for (size_t paddingHeight = 0; paddingHeight <= 1; paddingHeight++) {
        for (size_t paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .paddingHeight(paddingHeight)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_Mx1_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 3)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .strideHeight(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_Mx1_pool_with_dilation) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2 * poolSize)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .dilationHeight(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_pool_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_pool_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_pool_with_qmin) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmin(192)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmin(192)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_small_pool_with_qmax) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmax(192)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmax(192)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_1xM_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_1xM_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 3; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      for (size_t paddingWidth = 0; paddingWidth <= 1; paddingWidth++) {
        for (size_t paddingRight = 0; paddingRight <= 1; paddingRight++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .paddingWidth(paddingWidth)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_1xM_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 4)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .strideWidth(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_1xM_pool_with_dilation) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(2 * poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .dilationWidth(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_Mx1_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_Mx1_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      for (size_t paddingHeight = 0; paddingHeight <= 1; paddingHeight++) {
        for (size_t paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .paddingHeight(paddingHeight)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_Mx1_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 3)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .strideHeight(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_Mx1_pool_with_dilation) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2 * poolSize)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .dilationHeight(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_pool_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_pool_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_pool_with_qmin) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmin(192)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmin(192)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_many_channels_large_pool_with_qmax) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmax(192)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmax(192)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_1xM_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_1xM_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 3; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      for (size_t paddingWidth = 0; paddingWidth <= 1; paddingWidth++) {
        for (size_t paddingRight = 0; paddingRight <= 1; paddingRight++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .paddingWidth(paddingWidth)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_1xM_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 4)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .strideWidth(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_1xM_pool_with_dilation) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(2 * poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .dilationWidth(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_Mx1_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_Mx1_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      for (size_t paddingHeight = 0; paddingHeight <= 1; paddingHeight++) {
        for (size_t paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .paddingHeight(paddingHeight)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_Mx1_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 3)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .strideHeight(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_Mx1_pool_with_dilation) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2 * poolSize)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .dilationHeight(2)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_with_qmin) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmin(192)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmin(192)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, unit_batch_few_channels_with_qmax) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmax(192)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmax(192)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_small_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_small_pool_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_small_pool_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_large_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr + 1; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_large_pool_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 5) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr + 1; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_many_channels_large_pool_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.u8maxpool.kr;
       channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
       channels += 5) {
    for (size_t poolSize = pytorch_qnnp_params.u8maxpool.mr + 1; poolSize <=
         pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_few_channels) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize++) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_few_channels_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize += 3) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, small_batch_few_channels_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
         poolSize += 3) {
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
          .testU8();
    }
  }
}

TEST(MAX_POOLING_OP, setup_increasing_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  MaxPoolingOperatorTester()
      .batchSize(3)
      .nextBatchSize(5)
      .inputHeight(8)
      .inputWidth(8)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupU8();
}

TEST(MAX_POOLING_OP, setup_decreasing_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  MaxPoolingOperatorTester()
      .batchSize(5)
      .nextBatchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupU8();
}

TEST(MAX_POOLING_OP, setup_changing_height) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  MaxPoolingOperatorTester()
      .batchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .nextInputHeight(9)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupU8();
  MaxPoolingOperatorTester()
      .batchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .nextInputHeight(7)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupU8();
}

TEST(MAX_POOLING_OP, setup_changing_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  MaxPoolingOperatorTester()
      .batchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .nextInputWidth(9)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupU8();
  MaxPoolingOperatorTester()
      .batchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .nextInputWidth(7)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupU8();
}

TEST(MAX_POOLING_OP, setup_swap_height_and_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  MaxPoolingOperatorTester()
      .batchSize(3)
      .inputHeight(9)
      .inputWidth(8)
      .nextInputHeight(8)
      .nextInputWidth(9)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupU8();
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
- `max-pooling-operator-tester.h`
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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/max-pooling.cc
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

- **File Documentation**: `max-pooling.cc_docs.md`
- **Keyword Index**: `max-pooling.cc_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/max-pooling.cc_docs.md
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

- **File Documentation**: `max-pooling.cc_docs.md_docs.md`
- **Keyword Index**: `max-pooling.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
