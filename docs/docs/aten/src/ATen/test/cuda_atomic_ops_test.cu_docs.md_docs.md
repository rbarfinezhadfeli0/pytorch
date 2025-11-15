# Documentation: `docs/aten/src/ATen/test/cuda_atomic_ops_test.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/cuda_atomic_ops_test.cu_docs.md`
- **Size**: 9,247 bytes (9.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/cuda_atomic_ops_test.cu`

## File Metadata

- **Path**: `aten/src/ATen/test/cuda_atomic_ops_test.cu`
- **Size**: 6,549 bytes (6.40 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cuda
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

This file is part of the PyTorch framework located at `aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/cuda/Atomic.cuh`
- `c10/test/util/Macros.h`
- `ATen/cuda/CUDAContext.h`
- `c10/cuda/CUDAException.h`
- `cmath`


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
python aten/src/ATen/test/cuda_atomic_ops_test.cu
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `cuda_atomic_ops_test.cu_docs.md`
- **Keyword Index**: `cuda_atomic_ops_test.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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
python docs/aten/src/ATen/test/cuda_atomic_ops_test.cu_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cuda_atomic_ops_test.cu_docs.md_docs.md`
- **Keyword Index**: `cuda_atomic_ops_test.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
