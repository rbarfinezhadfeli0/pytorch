# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle.cc_docs.md`
- **Size**: 10,064 bytes (9.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle.cc`
- **Size**: 7,404 bytes (7.23 KB)
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

#include "channel-shuffle-operator-tester.h"

TEST(CHANNEL_SHUFFLE_OP, zero_batch) {
  ChannelShuffleOperatorTester()
      .batchSize(0)
      .groups(2)
      .groupChannels(4)
      .iterations(1)
      .testX8();
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_unit_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(1)
        .groups(2)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, three_groups_unit_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(1)
        .groups(3)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_unit_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(1)
        .groups(4)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_unit_batch) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(1)
          .groups(groups)
          .groupChannels(groupChannels)
          .iterations(3)
          .testX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_small_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(2)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, three_groups_small_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(3)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_small_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(4)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_small_batch) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(3)
          .groups(groups)
          .groupChannels(groupChannels)
          .iterations(3)
          .testX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_small_batch_with_input_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(2)
        .groupChannels(groupChannels)
        .inputStride(511)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, three_groups_small_batch_with_input_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(3)
        .groupChannels(groupChannels)
        .inputStride(511)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_small_batch_with_input_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(4)
        .groupChannels(groupChannels)
        .inputStride(511)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_small_batch_with_input_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(3)
          .groups(groups)
          .groupChannels(groupChannels)
          .inputStride(1007)
          .iterations(3)
          .testX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_small_batch_with_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(2)
        .groupChannels(groupChannels)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, three_groups_small_batch_with_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(3)
        .groupChannels(groupChannels)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_small_batch_with_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(4)
        .groupChannels(groupChannels)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_small_batch_with_output_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(3)
          .groups(groups)
          .groupChannels(groupChannels)
          .outputStride(1111)
          .iterations(3)
          .testX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_small_batch_with_input_and_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(2)
        .groupChannels(groupChannels)
        .inputStride(511)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(
    CHANNEL_SHUFFLE_OP,
    three_groups_small_batch_with_input_and_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(3)
        .groupChannels(groupChannels)
        .inputStride(511)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_small_batch_with_input_and_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(4)
        .groupChannels(groupChannels)
        .inputStride(511)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_small_batch_with_input_and_output_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(3)
          .groups(groups)
          .groupChannels(groupChannels)
          .inputStride(1007)
          .outputStride(1111)
          .iterations(3)
          .testX8();
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
- `channel-shuffle-operator-tester.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle.cc
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

- **File Documentation**: `channel-shuffle.cc_docs.md`
- **Keyword Index**: `channel-shuffle.cc_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle.cc_docs.md
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
- [`q8vadd.cc_kw.md_docs.md`](./q8vadd.cc_kw.md_docs.md)
- [`global-average-pooling.cc_docs.md_docs.md`](./global-average-pooling.cc_docs.md_docs.md)


## Cross-References

- **File Documentation**: `channel-shuffle.cc_docs.md_docs.md`
- **Keyword Index**: `channel-shuffle.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
