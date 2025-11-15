# Documentation: cuda_atomic_ops_test.cu

## File Metadata
- **Path**: `aten/src/ATen/test/cuda_atomic_ops_test.cu`
- **Size**: 6549 bytes
- **Lines**: 252
- **Extension**: .cu
- **Type**: Regular file

## Original Source

```cu
#include <gtest/gtest.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/test/util/Macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>

constexpr int blocksize = 256;
constexpr int factor = 4;
constexpr int arraysize = blocksize / factor;

template <typename T>
__global__ void addition_test_kernel(T * a, T * sum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (tid) % arraysize;

  gpuAtomicAdd(&sum[idx], a[idx]);
}

template <typename T>
__global__ void mul_test_kernel(T * a, T * sum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (tid) % arraysize;

  gpuAtomicMul(&sum[idx], a[idx]);
}

template <typename T>
__global__ void max_test_kernel(T * a, T * max) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int a_idx = (tid) % (arraysize * factor);
  int idx = a_idx / factor;

  gpuAtomicMax(&max[idx], a[a_idx]);
}

template <typename T>
__global__ void min_test_kernel(T * a, T * min) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int a_idx = (tid) % (arraysize * factor);
  int idx = a_idx / factor;

  gpuAtomicMin(&min[idx], a[a_idx]);
}

template <typename T>
void test_atomic_add() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *ad, *sumd;

  std::vector<T> a(arraysize);
  std::vector<T> sum(arraysize);
  std::vector<T> answer(arraysize);

  for (int i = 0; i < arraysize; ++i) {
    a[i] = 1;
    sum[i] = 0;
    answer[i] = factor;
  }

  cudaMalloc((void**)&ad, arraysize * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);

  addition_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(sum.data(), sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    ASSERT_EQ(sum[i], answer[i]) << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
}

template <typename T>
void test_atomic_mul() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *ad, *sumd;

  std::vector<T> a(arraysize);
  std::vector<T> sum(arraysize);
  std::vector<T> answer(arraysize);

  for (int i = 0; i < arraysize; ++i) {
    a[i] = 2;
    sum[i] = 2;
    answer[i] = pow(sum[i], static_cast<T>(factor + 1));
  }

  cudaMalloc((void**)&ad, arraysize * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);

  mul_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(sum.data(), sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    ASSERT_EQ(sum[i], answer[i]) << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
}

template <typename T>
void test_atomic_max() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *ad, *sumd;

  std::vector<T> a(arraysize * factor);
  std::vector<T> sum(arraysize);
  std::vector<T> answer(arraysize);

  int j;
  for (int i = 0; i < arraysize * factor; ++i) {
    a[i] = i;
    if (i % factor == 0) {
      j = i / factor;
      sum[j] = std::numeric_limits<T>::lowest();
      answer[j] = (j + 1) * factor - 1;
    }
  }

  cudaMalloc((void**)&ad, arraysize * factor * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a.data(), arraysize * factor * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);

  max_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(sum.data(), sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    ASSERT_EQ(sum[i], answer[i]) << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
}

template <typename T>
void test_atomic_min() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *ad, *sumd;

  std::vector<T> a(arraysize * factor);
  std::vector<T> sum(arraysize);
  std::vector<T> answer(arraysize);

  int j;
  for (int i = 0; i < arraysize * factor; ++i) {
    a[i] = i;
    if (i % factor == 0) {
      j = i / factor;
      sum[j] = std::numeric_limits<T>::max();
      answer[j] = j * factor;
    }
  }

  cudaMalloc((void**)&ad, arraysize * factor * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a.data(), arraysize * factor * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);

  min_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(sum.data(), sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    ASSERT_EQ(sum[i], answer[i]) << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
}

TEST(TestAtomicOps, TestAtomicAdd) {
  if (!at::cuda::is_available()) return;
  test_atomic_add<uint8_t>();
  test_atomic_add<int8_t>();
  test_atomic_add<int16_t>();
  test_atomic_add<int32_t>();
  test_atomic_add<int64_t>();

  test_atomic_add<at::BFloat16>();
  test_atomic_add<at::Half>();
  test_atomic_add<float>();
  test_atomic_add<double>();
  test_atomic_add<c10::complex<float> >();
  test_atomic_add<c10::complex<double> >();
}

TEST(TestAtomicOps, DISABLED_ON_WINDOWS(TestAtomicMul)) {
  if (!at::cuda::is_available()) return;
  test_atomic_mul<uint8_t>();
  test_atomic_mul<int8_t>();
  test_atomic_mul<int16_t>();
  test_atomic_mul<int32_t>();
  test_atomic_mul<int64_t>();
  test_atomic_mul<at::BFloat16>();
  test_atomic_mul<at::Half>();
  test_atomic_mul<float>();
  test_atomic_mul<double>();
}

TEST(TestAtomicOps, DISABLED_ON_WINDOWS(TestAtomicMax)) {
  if (!at::cuda::is_available()) return;
  test_atomic_max<uint8_t>();
  test_atomic_max<int8_t>();
  test_atomic_max<int16_t>();
  test_atomic_max<int32_t>();
  test_atomic_max<int64_t>();
  test_atomic_max<at::BFloat16>();
  test_atomic_max<at::Half>();
  test_atomic_max<float>();
  test_atomic_max<double>();
}

TEST(TestAtomicOps, DISABLED_ON_WINDOWS(TestAtomicMin)) {
  if (!at::cuda::is_available()) return;
  test_atomic_min<uint8_t>();
  test_atomic_min<int8_t>();
  test_atomic_min<int16_t>();
  test_atomic_min<int32_t>();
  test_atomic_min<int64_t>();
  test_atomic_min<at::BFloat16>();
  test_atomic_min<at::Half>();
  test_atomic_min<float>();
  test_atomic_min<double>();
}

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough


## Key Components

The file contains 642 words across 252 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 6549 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
