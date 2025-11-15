# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-operator-tester.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-operator-tester.h_docs.md`
- **Size**: 15,696 bytes (15.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-operator-tester.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-operator-tester.h`
- **Size**: 12,678 bytes (12.38 KB)
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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#include <memory>

#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <qnnpack/AlignedAllocator.h>

class FullyConnectedOperatorTester {
 public:
  inline FullyConnectedOperatorTester& inputChannels(size_t inputChannels) {
    assert(inputChannels >= 1);
    this->inputChannels_ = inputChannels;
    return *this;
  }

  inline size_t inputChannels() const {
    return this->inputChannels_;
  }

  inline FullyConnectedOperatorTester& outputChannels(size_t outputChannels) {
    assert(outputChannels >= 1);
    this->outputChannels_ = outputChannels;
    return *this;
  }

  inline size_t outputChannels() const {
    return this->outputChannels_;
  }

  inline FullyConnectedOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline FullyConnectedOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride >= 1);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return inputChannels();
    } else {
      assert(this->inputStride_ >= inputChannels());
      return this->inputStride_;
    }
  }

  inline FullyConnectedOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride >= 1);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return outputChannels();
    } else {
      assert(this->outputStride_ >= outputChannels());
      return this->outputStride_;
    }
  }

  inline FullyConnectedOperatorTester& per_channel(bool per_channel) {
    this->per_channel_ = per_channel;
    return *this;
  }

  inline bool per_channel() const {
    return this->per_channel_;
  }

  inline FullyConnectedOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline FullyConnectedOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline FullyConnectedOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  enum class Mode {
    Static,
    Dynamic,
    Runtime,
  };

  void testQ8(const Mode mode) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    auto f32rng =
        std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    std::vector<uint8_t> input(
        (batchSize() - 1) * inputStride() + inputChannels() + 8);
    std::vector<uint8_t> kernel(outputChannels() * inputChannels());
    std::vector<int32_t> bias(outputChannels());
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + outputChannels());
    std::vector<float> output_dynamic(output.size());
    std::vector<int32_t> accumulators(batchSize() * outputChannels());

    const uint8_t* const inputPtr = input.data() + 8;
    const uint8_t inputZeroPoint = 127;
    // Make number of output channels multiple of 8.
    // This is the least common denominator for SSE/ARM kernels we have.
    size_t num_zero_points_padded = outputChannels() + 8;
    std::vector<uint8_t> kernelZeroPoints(num_zero_points_padded, 127);

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      if (per_channel()) {
        std::generate(kernelZeroPoints.begin(), kernelZeroPoints.end(), std::ref(u8rng));
      }
      std::fill(output.begin(), output.end(), 0xA5);
      std::fill(output_dynamic.begin(), output_dynamic.end(), 0.0f);
      std::fill(accumulators.begin(), accumulators.end(), 0);

      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oc = 0; oc < outputChannels(); oc++) {
          accumulators[i * outputChannels() + oc] = bias[oc];
        }
      }
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oc = 0; oc < outputChannels(); oc++) {
          for (size_t ic = 0; ic < inputChannels(); ic++) {
            accumulators[i * outputChannels() + oc] +=
                (int32_t(inputPtr[i * inputStride() + ic]) -
                 int32_t(inputZeroPoint)) *
                (int32_t(kernel[oc * inputChannels() + ic]) -
                 int32_t(kernelZeroPoints[oc]));
          }
        }
      }

      // Create dummy min/max for empty inputs.
      // These are only used to compute scale and zero point,
      // and real callers will just pull those values from the model.
      const int32_t accumulatorsMin = accumulators.empty()
          ? 0
          : *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulatorsMax = accumulators.empty()
          ? 900
          : *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double outputScale =
          double(uint32_t(accumulatorsMax - accumulatorsMin)) / 255.0;
      const uint8_t outputZeroPoint = uint8_t(std::max(
          std::min(
              lrint(
                  127.5 -
                  0.5 * double(accumulatorsMin + accumulatorsMax) /
                      outputScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      // 1 bcz input_scale and kernel_scale are both 1.
      std::vector<float>
        requantization_scales(num_zero_points_padded, 1.0 * 1.0 / outputScale);
      if (per_channel()) {
        auto scale_generator = [&]() -> float {return (f32rng()/outputScale);};
        std::generate(
            requantization_scales.begin(),
            requantization_scales.end(),
            std::ref(scale_generator));
      }

      switch(mode) {
        case Mode::Static:
        {
          pytorch_qnnp_operator_t convolution = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_fully_connected_nc_q8(
                  inputChannels(),
                  outputChannels(),
                  inputZeroPoint,
                  kernelZeroPoints.data(),
                  kernel.data(),
                  bias.data(),
                  outputZeroPoint,
                  qmin(),
                  qmax(),
                  0,
                  requantization_scales.data(),
                  &convolution));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_fully_connected_nc_q8(
                  convolution,
                  batchSize(),
                  inputPtr,
                  inputStride(),
                  output.data(),
                  outputStride()));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(convolution, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_delete_operator(convolution));
          convolution = nullptr;
        }
        break;

        case Mode::Dynamic:
        {
          auto packW = std::unique_ptr<qnnpack::PackBMatrix>(
              new qnnpack::PackBMatrix(
                  inputChannels(),
                  outputChannels(),
                  kernelZeroPoints.data(),
                  requantization_scales.data(),
                  kernel.data(),
                  nullptr));

          // Attention! Bias size must be a multiple of 8.
          constexpr size_t kBiasSizeMultiple = 8u;
          std::vector<float, AlignedAllocator<float, 32>> bias_float(
              (bias.size() + (kBiasSizeMultiple - 1)) & -kBiasSizeMultiple);
          std::copy(bias.cbegin(), bias.cend(), bias_float.begin());

          const pytorch_qnnp_status runStatus = qnnpack::qnnpackLinearDynamic(
              batchSize() /* batch_size */,
              inputChannels() /* input_channels */,
              outputChannels() /* output_channels */,
              inputZeroPoint,
              kernelZeroPoints.data(),
              requantization_scales.data(), /* Dequantization scale */
              inputPtr,
              inputChannels() /* input_stride */,
              packW->getPackedWeights(),
              bias_float.data(),
              output_dynamic.data(),
              outputStride() /* output_stride */,
              nullptr /* threadpool */);
          ASSERT_EQ(pytorch_qnnp_status_success, runStatus);
        }
        break;

        case Mode::Runtime:
        {
          auto packW = std::unique_ptr<qnnpack::PackBMatrix>(
              new qnnpack::PackBMatrix(
                  inputChannels(),
                  outputChannels(),
                  kernelZeroPoints.data(),
                  requantization_scales.data(),
                  kernel.data(),
                  bias.data()));

          const pytorch_qnnp_status runStatus = qnnpack::qnnpackLinear(
              batchSize() /* batch_size */,
              inputChannels() /* input_channels */,
              outputChannels() /* output_channels */,
              inputZeroPoint,
              kernelZeroPoints.data(),
              requantization_scales.data(),
              outputZeroPoint,
              qmin(),
              qmax(),
              inputPtr,
              inputChannels() /* input_stride */,
              packW->getPackedWeights(),
              output.data(),
              outputStride() /* output_stride */,
              nullptr /* threadpool */);
          ASSERT_EQ(pytorch_qnnp_status_success, runStatus);
        }
        break;

        default:
          // Undefined!
          ASSERT_TRUE(false);
      }

      switch (mode) {
        case Mode::Static:
        case Mode::Runtime:
        {
          for (size_t i = 0; i < batchSize(); i++) {
            for (size_t c = 0; c < outputChannels(); c++) {
              const double scaledAccumulator =
                  accumulators[i * outputChannels() + c] *
                  requantization_scales[c];
              const double clampedAccumulator = std::max(
                  std::min(
                      scaledAccumulator, double(qmax()) - double(outputZeroPoint)),
                  double(qmin()) - double(outputZeroPoint));
              ASSERT_NEAR(
                  clampedAccumulator,
                  (int32_t(output[i * outputStride() + c]) - outputZeroPoint),
                  0.9)
                  << "batch index = " << i << ", channel = " << c;
            }
          }
        }
        break;

        case Mode::Dynamic:
        {
          // Bias is added post scaling, as float.
          for (size_t i = 0; i < batchSize(); i++) {
            for (size_t oc = 0; oc < outputChannels(); oc++) {
              accumulators[i * outputChannels() + oc] -= bias[oc];
            }
          }
          for (size_t i = 0; i < batchSize(); i++) {
            for (size_t c = 0; c < outputChannels(); c++) {
              const float ref = ((float)accumulators[i * outputChannels() + c] *
                           requantization_scales[c]) +
                  float(bias[c]);
              ASSERT_NEAR(
                  output_dynamic[i * outputChannels() + c],
                  ref,
                  std::abs(ref) * 1.0e-4)
                  << "at " << i << ", " << c << ": reference = " << ref
                  << ", optimized = "
                  << output_dynamic[i * outputChannels() + c];
            }
          }
        }
        break;

        default:
          // Undefined!
          ASSERT_TRUE(false);
      }
    }
  }

 private:
  size_t inputChannels_{1};
  size_t inputStride_{0};
  size_t outputChannels_{1};
  size_t outputStride_{0};
  size_t batchSize_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
  bool per_channel_{false};
};

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `FullyConnectedOperatorTester`, `Mode`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `algorithm`
- `cmath`
- `cstddef`
- `cstdlib`
- `functional`
- `random`
- `vector`
- `memory`
- `pytorch_qnnpack.h`
- `qnnpack_func.h`
- `qnnpack/AlignedAllocator.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-operator-tester.h
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

- **File Documentation**: `fully-connected-operator-tester.h_docs.md`
- **Keyword Index**: `fully-connected-operator-tester.h_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-operator-tester.h_docs.md
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

- **File Documentation**: `fully-connected-operator-tester.h_docs.md_docs.md`
- **Keyword Index**: `fully-connected-operator-tester.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
