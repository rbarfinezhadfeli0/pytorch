# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/vadd-microkernel-tester.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/vadd-microkernel-tester.h_docs.md`
- **Size**: 8,726 bytes (8.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/vadd-microkernel-tester.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/vadd-microkernel-tester.h`
- **Size**: 5,868 bytes (5.73 KB)
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

#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

class VAddMicrokernelTester {
 public:
  inline VAddMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline VAddMicrokernelTester& inplaceA(bool inplaceA) {
    this->inplaceA_ = inplaceA;
    return *this;
  }

  inline bool inplaceA() const {
    return this->inplaceA_;
  }

  inline VAddMicrokernelTester& inplaceB(bool inplaceB) {
    this->inplaceB_ = inplaceB;
    return *this;
  }

  inline bool inplaceB() const {
    return this->inplaceB_;
  }

  inline VAddMicrokernelTester& aScale(float aScale) {
    assert(aScale > 0.0f);
    assert(std::isnormal(aScale));
    this->aScale_ = aScale;
    return *this;
  }

  inline float aScale() const {
    return this->aScale_;
  }

  inline VAddMicrokernelTester& aZeroPoint(uint8_t aZeroPoint) {
    this->aZeroPoint_ = aZeroPoint;
    return *this;
  }

  inline uint8_t aZeroPoint() const {
    return this->aZeroPoint_;
  }

  inline VAddMicrokernelTester& bScale(float bScale) {
    assert(bScale > 0.0f);
    assert(std::isnormal(bScale));
    this->bScale_ = bScale;
    return *this;
  }

  inline float bScale() const {
    return this->bScale_;
  }

  inline VAddMicrokernelTester& bZeroPoint(uint8_t bZeroPoint) {
    this->bZeroPoint_ = bZeroPoint;
    return *this;
  }

  inline uint8_t bZeroPoint() const {
    return this->bZeroPoint_;
  }

  inline VAddMicrokernelTester& yScale(float yScale) {
    assert(yScale > 0.0f);
    assert(std::isnormal(yScale));
    this->yScale_ = yScale;
    return *this;
  }

  inline float yScale() const {
    return this->yScale_;
  }

  inline VAddMicrokernelTester& yZeroPoint(uint8_t yZeroPoint) {
    this->yZeroPoint_ = yZeroPoint;
    return *this;
  }

  inline uint8_t yZeroPoint() const {
    return this->yZeroPoint_;
  }

  inline VAddMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VAddMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VAddMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_q8vadd_ukernel_function q8vadd) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a(n());
    std::vector<uint8_t> b(n());
    std::vector<uint8_t> y(n());
    std::vector<float> yFP(n());
    std::vector<uint8_t> yRef(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      if (inplaceA() || inplaceB()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const uint8_t* aData = inplaceA() ? y.data() : a.data();
      const uint8_t* bData = inplaceB() ? y.data() : b.data();

      /* Prepare quantization parameters */
      const union pytorch_qnnp_add_quantization_params quantizationParams =
          pytorch_qnnp_compute_add_quantization_params(
              aZeroPoint(),
              bZeroPoint(),
              yZeroPoint(),
              aScale() / yScale(),
              bScale() / yScale(),
              qmin(),
              qmax());
      const union pytorch_qnnp_add_quantization_params
          scalarQuantizationParams =
              pytorch_qnnp_compute_scalar_add_quantization_params(
                  aZeroPoint(),
                  bZeroPoint(),
                  yZeroPoint(),
                  aScale() / yScale(),
                  bScale() / yScale(),
                  qmin(),
                  qmax());

      /* Compute reference results */
      for (size_t i = 0; i < n(); i++) {
        yFP[i] = float(yZeroPoint()) +
            float(int32_t(aData[i]) - int32_t(aZeroPoint())) *
                (aScale() / yScale()) +
            float(int32_t(bData[i]) - int32_t(bZeroPoint())) *
                (bScale() / yScale());
        yFP[i] = std::min<float>(yFP[i], float(qmax()));
        yFP[i] = std::max<float>(yFP[i], float(qmin()));
        yRef[i] = pytorch_qnnp_add_quantize(
            aData[i], bData[i], scalarQuantizationParams);
      }

      /* Call optimized micro-kernel */
      q8vadd(n(), aData, bData, y.data(), &quantizationParams);

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))
            << "at " << i << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))
            << "at " << i << ", n = " << n();
        ASSERT_NEAR(float(int32_t(y[i])), yFP[i], 0.6f)
            << "at " << i << ", n = " << n();
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))
            << "at " << i << ", n = " << n();
      }
    }
  }

 private:
  size_t n_{1};
  bool inplaceA_{false};
  bool inplaceB_{false};
  float aScale_{0.75f};
  float bScale_{1.25f};
  float yScale_{0.96875f};
  uint8_t aZeroPoint_{121};
  uint8_t bZeroPoint_{127};
  uint8_t yZeroPoint_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `VAddMicrokernelTester`


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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/vadd-microkernel-tester.h
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

- **File Documentation**: `vadd-microkernel-tester.h_docs.md`
- **Keyword Index**: `vadd-microkernel-tester.h_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/vadd-microkernel-tester.h_docs.md
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

- **File Documentation**: `vadd-microkernel-tester.h_docs.md_docs.md`
- **Keyword Index**: `vadd-microkernel-tester.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
