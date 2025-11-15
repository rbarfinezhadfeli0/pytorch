# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle-operator-tester.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle-operator-tester.h_docs.md`
- **Size**: 7,109 bytes (6.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle-operator-tester.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle-operator-tester.h`
- **Size**: 4,253 bytes (4.15 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```c
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

class ChannelShuffleOperatorTester {
 public:
  inline ChannelShuffleOperatorTester& groups(size_t groups) {
    assert(groups != 0);
    this->groups_ = groups;
    return *this;
  }

  inline size_t groups() const {
    return this->groups_;
  }

  inline ChannelShuffleOperatorTester& groupChannels(size_t groupChannels) {
    assert(groupChannels != 0);
    this->groupChannels_ = groupChannels;
    return *this;
  }

  inline size_t groupChannels() const {
    return this->groupChannels_;
  }

  inline size_t channels() const {
    return groups() * groupChannels();
  }

  inline ChannelShuffleOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return channels();
    } else {
      assert(this->inputStride_ >= channels());
      return this->inputStride_;
    }
  }

  inline ChannelShuffleOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return channels();
    } else {
      assert(this->outputStride_ >= channels());
      return this->outputStride_;
    }
  }

  inline ChannelShuffleOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline ChannelShuffleOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testX8() const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Create, setup, run, and destroy Channel Shuffle operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t channel_shuffle_op = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_channel_shuffle_nc_x8(
              groups(), groupChannels(), 0, &channel_shuffle_op));
      ASSERT_NE(nullptr, channel_shuffle_op);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_channel_shuffle_nc_x8(
              channel_shuffle_op,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(
              channel_shuffle_op, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(channel_shuffle_op));
      channel_shuffle_op = nullptr;

      /* Verify results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t g = 0; g < groups(); g++) {
          for (size_t c = 0; c < groupChannels(); c++) {
            ASSERT_EQ(
                uint32_t(input[i * inputStride() + g * groupChannels() + c]),
                uint32_t(output[i * outputStride() + c * groups() + g]));
          }
        }
      }
    }
  }

 private:
  size_t groups_{1};
  size_t groupChannels_{1};
  size_t batchSize_{1};
  size_t inputStride_{0};
  size_t outputStride_{0};
  size_t iterations_{15};
};

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `ChannelShuffleOperatorTester`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `algorithm`
- `cassert`
- `cstddef`
- `cstdlib`
- `functional`
- `random`
- `vector`
- `pytorch_qnnpack.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle-operator-tester.h
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


## Cross-References

- **File Documentation**: `channel-shuffle-operator-tester.h_docs.md`
- **Keyword Index**: `channel-shuffle-operator-tester.h_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle-operator-tester.h_docs.md
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

- **File Documentation**: `channel-shuffle-operator-tester.h_docs.md_docs.md`
- **Keyword Index**: `channel-shuffle-operator-tester.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
