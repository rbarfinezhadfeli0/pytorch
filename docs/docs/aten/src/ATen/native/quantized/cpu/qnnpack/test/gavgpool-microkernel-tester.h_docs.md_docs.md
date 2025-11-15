# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/gavgpool-microkernel-tester.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/gavgpool-microkernel-tester.h_docs.md`
- **Size**: 11,753 bytes (11.48 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/gavgpool-microkernel-tester.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/gavgpool-microkernel-tester.h`
- **Size**: 8,830 bytes (8.62 KB)
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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

class GAvgPoolMicrokernelTester {
 public:
  inline GAvgPoolMicrokernelTester& m(size_t m) {
    assert(m != 0);
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GAvgPoolMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GAvgPoolMicrokernelTester& nr(size_t nr) {
    assert(nr != 0);
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline size_t packedN() const {
    return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
  }

  inline GAvgPoolMicrokernelTester& xStride(size_t xStride) {
    assert(xStride != 0);
    this->xStride_ = xStride;
    return *this;
  }

  inline size_t xStride() const {
    if (this->xStride_ == 0) {
      return n();
    } else {
      assert(this->xStride_ >= n());
      return this->xStride_;
    }
  }

  inline GAvgPoolMicrokernelTester& xScale(float xScale) {
    assert(xScale > 0.0f);
    assert(std::isnormal(xScale));
    this->xScale_ = xScale;
    return *this;
  }

  inline float xScale() const {
    return this->xScale_;
  }

  inline GAvgPoolMicrokernelTester& xZeroPoint(uint8_t xZeroPoint) {
    this->xZeroPoint_ = xZeroPoint;
    return *this;
  }

  inline uint8_t xZeroPoint() const {
    return this->xZeroPoint_;
  }

  inline GAvgPoolMicrokernelTester& yScale(float yScale) {
    assert(yScale > 0.0f);
    assert(std::isnormal(yScale));
    this->yScale_ = yScale;
    return *this;
  }

  inline float yScale() const {
    return this->yScale_;
  }

  inline GAvgPoolMicrokernelTester& yZeroPoint(uint8_t yZeroPoint) {
    this->yZeroPoint_ = yZeroPoint;
    return *this;
  }

  inline uint8_t yZeroPoint() const {
    return this->yZeroPoint_;
  }

  inline GAvgPoolMicrokernelTester& yMin(uint8_t yMin) {
    this->yMin_ = yMin;
    return *this;
  }

  inline uint8_t yMin() const {
    return this->yMin_;
  }

  inline GAvgPoolMicrokernelTester& yMax(uint8_t yMax) {
    this->yMax_ = yMax;
    return *this;
  }

  inline uint8_t yMax() const {
    return this->yMax_;
  }

  inline GAvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_q8gavgpool_up_ukernel_function q8gavgpool) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> x((m() - 1) * xStride() + n());
    std::vector<uint8_t> zero(n());
    std::vector<uint8_t> y(n());
    std::vector<uint8_t> yRef(n());
    std::vector<float> yFP(n());
    std::vector<int32_t> yAcc(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      /* Prepare quantization parameters */
      const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
          pytorch_qnnp_compute_avgpool_quantization_params(
              -int32_t(xZeroPoint()) * int32_t(m()),
              xScale() / (yScale() * float(m())),
              yZeroPoint(),
              yMin(),
              yMax());
      const union pytorch_qnnp_avgpool_quantization_params
          scalarQuantizationParams =
              pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                  -int32_t(xZeroPoint()) * int32_t(m()),
                  xScale() / (yScale() * float(m())),
                  yZeroPoint(),
                  yMin(),
                  yMax());

      /* Compute reference results */
      for (size_t j = 0; j < n(); j++) {
        int32_t acc = scalarQuantizationParams.scalar.bias;
        for (size_t i = 0; i < m(); i++) {
          acc += x[i * xStride() + j];
        }
        yAcc[j] = acc;
        yRef[j] = pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
        yFP[j] = float(acc) * (xScale() / (yScale() * float(m()))) +
            float(yZeroPoint());
        yFP[j] = std::min<float>(yFP[j], float(yMax()));
        yFP[j] = std::max<float>(yFP[j], float(yMin()));
      }

      /* Call optimized micro-kernel */
      q8gavgpool(
          m(),
          n(),
          x.data(),
          xStride() * sizeof(uint8_t),
          zero.data(),
          y.data(),
          &quantizationParams);

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(yMax()))
            << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(yMin()))
            << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_NEAR(float(int32_t(y[i])), yFP[i], 0.5001f)
            << "at position " << i << ", m = " << m() << ", n = " << n()
            << ", acc = " << yAcc[i];
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))
            << "at position " << i << ", m = " << m() << ", n = " << n()
            << ", acc = " << yAcc[i];
      }
    }
  }

  void test(pytorch_q8gavgpool_mp_ukernel_function q8gavgpool) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> x((m() - 1) * xStride() + n());
    std::vector<int32_t, AlignedAllocator<int32_t, 16>> mpAcc(packedN());
    std::vector<uint8_t> zero(n());
    std::vector<uint8_t> y(n());
    std::vector<uint8_t> yRef(n());
    std::vector<float> yFP(n());
    std::vector<int32_t> yAcc(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      /* Prepare quantization parameters */
      const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
          pytorch_qnnp_compute_avgpool_quantization_params(
              -int32_t(xZeroPoint()) * int32_t(m()),
              xScale() / (yScale() * float(m())),
              yZeroPoint(),
              yMin(),
              yMax());
      const union pytorch_qnnp_avgpool_quantization_params
          scalarQuantizationParams =
              pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                  -int32_t(xZeroPoint()) * int32_t(m()),
                  xScale() / (yScale() * float(m())),
                  yZeroPoint(),
                  yMin(),
                  yMax());

      /* Compute reference results */
      for (size_t j = 0; j < n(); j++) {
        int32_t acc = scalarQuantizationParams.scalar.bias;
        for (size_t i = 0; i < m(); i++) {
          acc += x[i * xStride() + j];
        }

        yAcc[j] = acc;
        yRef[j] = pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
        yFP[j] = float(acc) * (xScale() / (yScale() * float(m()))) +
            float(yZeroPoint());
        yFP[j] = std::min<float>(yFP[j], float(yMax()));
        yFP[j] = std::max<float>(yFP[j], float(yMin()));
      }

      /* Call optimized micro-kernel */
      q8gavgpool(
          m(),
          n(),
          x.data(),
          xStride() * sizeof(uint8_t),
          zero.data(),
          mpAcc.data(),
          y.data(),
          &quantizationParams);

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(yMax()))
            << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(yMin()))
            << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_NEAR(float(int32_t(y[i])), yFP[i], 0.5001f)
            << "at position " << i << ", m = " << m() << ", n = " << n()
            << ", acc = " << yAcc[i];
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))
            << "at position " << i << ", m = " << m() << ", n = " << n()
            << ", acc = " << yAcc[i];
      }
    }
  }

 private:
  size_t m_{1};
  size_t n_{1};
  size_t nr_{1};
  size_t xStride_{0};
  float xScale_{1.25f};
  float yScale_{0.75f};
  uint8_t xZeroPoint_{121};
  uint8_t yZeroPoint_{133};
  uint8_t yMin_{0};
  uint8_t yMax_{255};
  size_t iterations_{15};
};

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `GAvgPoolMicrokernelTester`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `algorithm`
- `cassert`
- `cmath`
- `cstddef`
- `cstdlib`
- `functional`
- `random`
- `vector`
- `qnnpack/AlignedAllocator.h`
- `qnnpack/params.h`
- `qnnpack/requantization.h`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/gavgpool-microkernel-tester.h
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

- **File Documentation**: `gavgpool-microkernel-tester.h_docs.md`
- **Keyword Index**: `gavgpool-microkernel-tester.h_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/gavgpool-microkernel-tester.h_docs.md
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

- **File Documentation**: `gavgpool-microkernel-tester.h_docs.md_docs.md`
- **Keyword Index**: `gavgpool-microkernel-tester.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
