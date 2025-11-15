# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h_docs.md`
- **Size**: 18,534 bytes (18.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h`
- **Size**: 15,424 bytes (15.06 KB)
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

#include <fp16.h>

#include <pack_block_sparse.h>
#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

#define MAYBE_UNUSED __attribute__((unused))

namespace {
  void fillBlockSparseWeights(
      uint8_t* b,
      size_t N,
      size_t K,
      size_t row_block_size,
      size_t col_block_size,
      float sparsity,
      const uint8_t* zero_points) {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    std::bernoulli_distribution dist{sparsity};
    for (uint32_t n = 0; n < N ; n += row_block_size) {
      for (uint32_t k = 0; k < K; k += col_block_size) {
        if (dist(rng)) {
          for (uint32_t nb = 0; (nb < row_block_size) && (n + nb < N); ++nb) {
            for (uint32_t kb = 0; (kb < col_block_size) && (k + kb < K); ++kb) {
              *(b + (n + nb) * K + k + kb) = zero_points[n + nb];
            }
          }
        }
      }
    }
  }

  // Temp Debug utils that will be removed later
  MAYBE_UNUSED void printMatrix(const char* name, const uint8_t* a, const size_t M, const size_t N) {
    std::cout << "Matrix START:" << name << "...\n";
    for (uint32_t m = 0; m < M ; ++m) {
      for (uint32_t n = 0; n < N; n++) {
        std::cout << (const uint32_t)(*(a + m * N + n)) << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << "Matrix END...\n\n";
  }

  MAYBE_UNUSED void printMatrix(const char* name, const float* a, const size_t M, const size_t N) {
    std::cout << "Matrix START:" << name << "...\n";
    for (uint32_t m = 0; m < M ; ++m) {
      for (uint32_t n = 0; n < N; n++) {
        std::cout << (*(a + m * N + n)) << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << "Matrix END...\n\n";
  }

}

class GemmBlockSparseMicrokernelTester {
 public:
  inline GemmBlockSparseMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline GemmBlockSparseMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline GemmBlockSparseMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GemmBlockSparseMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GemmBlockSparseMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline GemmBlockSparseMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  inline GemmBlockSparseMicrokernelTester& rowBlockSize(size_t block_size) {
    this->rowBlockSize_ = block_size;
    return *this;
  }

  inline GemmBlockSparseMicrokernelTester& colBlockSize(size_t block_size) {
    this->colBlockSize_ = block_size;
    return *this;
  }

  inline GemmBlockSparseMicrokernelTester& sparsity(float s) {
    this->sparsity_ = s;
    return *this;
  }

  inline size_t ks() const {
    return this->ks_;
  }

  inline size_t rowBlockSize() const {
    return this->rowBlockSize_;
  }

  inline size_t colBlockSize() const {
    return this->colBlockSize_;
  }

  inline float sparsity() const {
    return this->sparsity_;
  }

  inline size_t biasN() const {
    return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
  }

  inline GemmBlockSparseMicrokernelTester& aStride(size_t aStride) {
    this->aStride_ = aStride;
    return *this;
  }

  inline size_t aStride() const {
    return this->aStride_ == 0 ? k() : this->aStride_;
  }

  inline GemmBlockSparseMicrokernelTester& cStride(size_t cStride) {
    this->cStride_ = cStride;
    return *this;
  }

  inline size_t cStride() const {
    return this->cStride_ == 0 ? n() : this->cStride_;
  }

  inline GemmBlockSparseMicrokernelTester& aZeroPoint(uint8_t aZeroPoint) {
    this->aZeroPoint_ = aZeroPoint;
    return *this;
  }

  inline uint8_t aZeroPoint() const {
    return this->aZeroPoint_;
  }

  inline GemmBlockSparseMicrokernelTester& bZeroPoint(uint8_t bZeroPoint) {
    this->bZeroPoint_ = bZeroPoint;
    return *this;
  }

  inline uint8_t bZeroPoint() const {
    return this->bZeroPoint_;
  }

  inline GemmBlockSparseMicrokernelTester& multiplier(const float multiplier) {
    this->multiplier_ = multiplier;
    return *this;
  }

  inline float multiplier() const {
    return this->multiplier_;
  }

  inline GemmBlockSparseMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GemmBlockSparseMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GemmBlockSparseMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_q8gemm_dq_sparse_ukernel_function qgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<float, AlignedAllocator<float, 32>> bias(std::max<size_t>(8, n()));
    std::vector<float> c((m() - 1) * cStride() + n());
    std::vector<float> acc(m() * n());

    const uint8_t* aPtr = a.data();

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0.0f);
      size_t num_zero_points_padded = n() + 8;
      std::vector<uint8_t> kernel_zero_points
        (num_zero_points_padded, bZeroPoint());
      std::generate(kernel_zero_points.begin(), kernel_zero_points.end(), std::ref(u8rng));

      // This loop to ensure the assert_ne on b mat does not fire.
      uint8_t max_elem, min_elem;
      do {
        std::generate(b.begin(), b.end(), std::ref(u8rng));
        fillBlockSparseWeights(
            b.data(),
            n(),
            k(),
            rowBlockSize(),
            colBlockSize(),
            sparsity(),
            kernel_zero_points.data());
        max_elem = *std::max_element(b.cbegin(), b.cend());
        min_elem = *std::min_element(b.cbegin(), b.cend());
      } while (max_elem == min_elem);

      std::unique_ptr<qnnpack::BCSRMatrix> bcsr_matrix =
          qnnpack::generateBlockCSRMatrix<uint32_t>(
              b.data(),
              n(),
              k(),
              rowBlockSize(),
              colBlockSize(),
              kernel_zero_points.data());

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      auto f32rng =
          std::bind(std::uniform_real_distribution<float>(1, 5), rng);
      std::vector<float> dequantization_scales(num_zero_points_padded);
      std::generate(
          dequantization_scales.begin(),
          dequantization_scales.end(),
          std::ref(f32rng));
      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kIndex = 0; kIndex < k(); kIndex++) {
            ASSERT_LT(mIndex * n() + nIndex, acc.size());
            ASSERT_LT(mIndex * k() + kIndex, a.size());
            acc[mIndex * n() + nIndex] +=
                (int32_t(aPtr[mIndex * aStride() + kIndex]) -
                 int32_t(aZeroPoint())) *
                (int32_t(b[nIndex * k() + kIndex]) - int32_t(kernel_zero_points[nIndex]));
          }
          acc[mIndex * n() + nIndex] =
            acc[mIndex * n() + nIndex] *
            dequantization_scales[nIndex] +
            bias[nIndex];
        }
      }

      const struct pytorch_qnnp_conv_dynamic_quantization_params quantizationParams{
        aZeroPoint(),
        kernel_zero_points.data(),
        dequantization_scales.data(),
      };

      qgemm(
          m(),
          n(),
          aPtr,
          aStride() * sizeof(uint8_t),
          bcsr_matrix->values.data(),
          static_cast<const uint32_t*>(bcsr_matrix->row_values_data_ptr()),
          static_cast<const uint32_t*>(bcsr_matrix->col_indices_data_ptr()),
          bias.data(),
          c.data(),
          cStride(),
          0,
          &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_EQ(
              c[mIndex * cStride() + nIndex],
              acc[mIndex * n() + nIndex])
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << acc[mIndex * n() + nIndex]
              << ", optimized = " << c[mIndex * cStride() + nIndex]
              << ", Mr x Nr = " << mr() << " x " << nr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  template <typename SPARSE_INDICES_DTYPE, typename GEMM_UKERNEL_DTYPE>
  void test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_function packa,
      GEMM_UKERNEL_DTYPE qgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<float, AlignedAllocator<float, 32>> bias(std::max<size_t>(8, n()));
    std::vector<float> c((m() - 1) * cStride() + n());
    std::vector<float> acc(m() * n());
    auto m_blocks = (m() + mr()  - 1) / mr();
    // While colBlockSize() is what kr is, we reuse 8x4/4x4 packing kernels
    // and thus a_packed has to be allocated accordingly.
    const uint32_t kr_value = 4;
    auto k_blocks = (k() + kr_value  - 1) / kr_value;
    std::vector<uint8_t> a_packed((m_blocks * k_blocks * mr() * kr_value) + 8, 0);

    const uint8_t* aPtr = a.data();

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0.0f);
      size_t num_zero_points_padded = n() + 8;
      std::vector<uint8_t> kernel_zero_points
        (num_zero_points_padded, bZeroPoint());

      uint8_t max_elem, min_elem;
      // This loop to ensure the assert_ne on b mat does not fire.
      do {
        std::generate(b.begin(), b.end(), std::ref(u8rng));
        fillBlockSparseWeights(
            b.data(),
            n(),
            k(),
            rowBlockSize(),
            colBlockSize(),
            sparsity(),
            kernel_zero_points.data());
        max_elem = *std::max_element(b.cbegin(), b.cend());
        min_elem = *std::min_element(b.cbegin(), b.cend());
      } while (max_elem == min_elem);
      std::unique_ptr<qnnpack::BCSRMatrix> bcsr_matrix =
          qnnpack::generateBlockCSRMatrix<SPARSE_INDICES_DTYPE>(
              b.data(),
              n(),
              k(),
              rowBlockSize(),
              colBlockSize(),
              kernel_zero_points.data());

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      auto f32rng =
          std::bind(std::uniform_real_distribution<float>(1, 5), rng);
      std::vector<float> dequantization_scales(num_zero_points_padded, 1.f);
      std::generate(
          dequantization_scales.begin(),
          dequantization_scales.end(),
          std::ref(f32rng));
      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kIndex = 0; kIndex < k(); kIndex++) {
            ASSERT_LT(mIndex * n() + nIndex, acc.size());
            ASSERT_LT(mIndex * k() + kIndex, a.size());
            acc[mIndex * n() + nIndex] +=
                (int32_t(aPtr[mIndex * aStride() + kIndex]) -
                 int32_t(aZeroPoint())) *
                (int32_t(b[nIndex * k() + kIndex]) - int32_t(kernel_zero_points[nIndex]));
          }
          acc[mIndex * n() + nIndex] =
            acc[mIndex * n() + nIndex] *
            dequantization_scales[nIndex] +
            bias[nIndex];
        }
      }

      const struct pytorch_qnnp_conv_dynamic_quantization_params quantizationParams{
        aZeroPoint(),
        kernel_zero_points.data(),
        dequantization_scales.data(),
      };

      packa(
          m(),
          k(),
          aPtr,
          aStride() * sizeof(uint8_t),
          a_packed.data()
          );

      qgemm(
          m(),
          n(),
          a_packed.data(),
          bcsr_matrix->values.data(),
          static_cast<const SPARSE_INDICES_DTYPE*>(
              bcsr_matrix->row_values_data_ptr()),
          static_cast<const SPARSE_INDICES_DTYPE*>(
              bcsr_matrix->col_indices_data_ptr()),
          bias.data(),
          c.data(),
          cStride(),
          0,
          &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_NEAR(
              c[mIndex * cStride() + nIndex],
              acc[mIndex * n() + nIndex],
              std::abs(acc[mIndex * n() + nIndex]) * 1.0e-3f)
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << acc[mIndex * n() + nIndex]
              << ", optimized = " << c[mIndex * cStride() + nIndex]
              << ", Mr x Nr = " << mr() << " x " << nr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t aStride_{0};
  size_t cStride_{0};
  size_t rowBlockSize_{1};
  size_t colBlockSize_{4};
  uint8_t aZeroPoint_{0};
  uint8_t bZeroPoint_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{10};
  float multiplier_{2.0f};
  float sparsity_{0.7f};
};

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `GemmBlockSparseMicrokernelTester`, `pytorch_qnnp_conv_dynamic_quantization_params`, `pytorch_qnnp_conv_dynamic_quantization_params`


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
- `fp16.h`
- `pack_block_sparse.h`
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
python aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h
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

- **File Documentation**: `gemm-block-sparse-microkernel-tester.h_docs.md`
- **Keyword Index**: `gemm-block-sparse-microkernel-tester.h_kw.md`
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
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h_docs.md
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

- **File Documentation**: `gemm-block-sparse-microkernel-tester.h_docs.md_docs.md`
- **Keyword Index**: `gemm-block-sparse-microkernel-tester.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
