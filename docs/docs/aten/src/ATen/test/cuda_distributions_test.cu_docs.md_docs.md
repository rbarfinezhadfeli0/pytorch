# Documentation: `docs/aten/src/ATen/test/cuda_distributions_test.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/cuda_distributions_test.cu_docs.md`
- **Size**: 9,887 bytes (9.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/cuda_distributions_test.cu`

## File Metadata

- **Path**: `aten/src/ATen/test/cuda_distributions_test.cu`
- **Size**: 7,194 bytes (7.03 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cuda
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Randperm.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

__global__ void expected_uniforms(float* x, uint64_t counter_offset) {
  for(int i=0; i < 4; i++) {
    curandStatePhilox4_32_10_t state;
    curand_init(
            123,
            i,
            counter_offset,
            &state);
    auto ret = curand_uniform4(&state);
    x[i] = ret.x;
  }
}

/**
 * Helper function that asserts call to uniform_ starts from the correct
 * philox offset.
 *   - Get 4 randoms with counter_offset for thread {0,1,2,3} from expected_uniforms
 *     kernel above.
 *   - Now get 4 more randoms from uniform_ (note thread {0,1,2,3} for this call should
 *     start from a counter_offset value)
 *   - the 4 randoms from expected_uniforms kernel and the 4 randoms from the previous call
 *     of uniform_ should match, signifying that the philox offset was
 *     incremented properly and no randoms are being reused from previous calls
 */
void assert_with_expected_uniforms(uint64_t counter_offset) {
  // allocate 4 float on host memory
  float *x;
  cudaMallocManaged(&x, 4*sizeof(float));

  // launch kernel to get expected randoms
  expected_uniforms<<<1, 1>>>(x, counter_offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // get 4 new float from uniform_()
  auto self = at::empty({4}, at::TensorOptions(at::kCUDA));
  self.uniform_();

  // check randoms from expected_uniforms kernel are equal to the randoms from the second
  // call of uniform_()
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(self[i].item().to<float>(), x[i]);
  }

  // Free memory
  cudaFree(x);
}

TEST(DistributionsTest, TestPhiloxIncrementSmallUniformTensor) {
  // Test Description:
  //   In Distributions.cu we mentioned that philox increment
  //   should be at least the number of curand() random numbers used in
  //   each thread. In this test, we make sure that uniform_ correctly
  //   increments philox and doesn't reuse randoms from previous calls
  //   for a small tensor size of 4.
  //    - We check that by first getting 4 randoms from uniform_.
  //      Once we get these 4 randoms, that would mean that philox counter for
  //      thread 0, 1, 2 and 3, was incremented by 4 (check calc_execution_policy
  //      function for details).
  //    - assert the call to uniform_ will start from counter_offset of 4

  // if cuda not available, return
  if (!at::cuda::is_available()) return;

  // manual seed to 123
  at::manual_seed(123);

  // get 4 randoms from uniform_(), philox offset is now incremented to 4 by this call
  at::empty({4}, at::TensorOptions(at::kCUDA)).uniform_();

  // expected uniforms will start from counter offset of 4
  assert_with_expected_uniforms(4);
}

TEST(DistributionsTest, TestPhiloxIncrementBigUniformTensor) {
  // Test Description:
  //   In Distributions.cu we mentioned that philox increment
  //   should be at least the number of curand() random numbers used in
  //   each thread. In this test, we make sure that uniform_ correctly
  //   increments philox and doesn't reuse randoms from previous calls
  //   for a big size tensor.
  //    - First of all, we come up with what the size of the big tensor
  //      should be for this test. Our goal is to show that when the uniform_
  //      kernel runs at full occupancy (i.e. when the number of elements is
  //      greater the number of threads launched), it hits the unroll loop in
  //      the uniform_ kernel.
  //    - Hence, we set the size of the tensor in this test to be 8 times the
  //      maximum number of threads we can launch. This means that, each thread
  //      will be yielding 8 elements, and as a result, curand_uniform4 will be
  //      called twice and all the 8 elements in a thread will consume all the
  //      float4 from the two calls of curand_uniform4 as a result of the unroll
  //      loop. Therefore, after this call to the uniform_, counter_offset for
  //      the next call to uniform_ will start from 8. This is what we test
  //      next.
  //    - assert that call to uniform_ will start from counter_offset of 8

  // if cuda not available, return
  if (!at::cuda::is_available()) return;

  // manual seed to 123
  at::manual_seed(123);

  // calculate maximum number of threads that can be launched
  // and set the numel to be 8 times that
  const int block_size = 256;
  uint32_t blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  dim3 grid(static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm);
  auto numel = block_size * grid.x * 8;

  // get numel randoms from uniform_(), philox offset is now incremented to 8 by this call
  at::empty({numel}, at::TensorOptions(at::kCUDA)).uniform_();

  // expected uniforms will start from counter offset of 8
  assert_with_expected_uniforms(8);
}

TEST(DistributionsTest, TestPhiloxIncrementSmallMultinomialTensor) {
  // Test Description:
  //   Same concept as TestPhiloxIncrementSmallUniformTensor.
  //   Multinomial increments offset by 4. Tests if uniform starts from the correct offset.

  // if cuda not available, return
  if (!at::cuda::is_available()) return;

  // manual seed to 123
  at::manual_seed(123);

  // get some multinomial samples
  // this will trigger torch.multinomial without replacement
  // which utilizes uniform which increments counter by 4.
  // num_samples in the following call is 4.
  at::ones({4}, at::TensorOptions(at::kCUDA)).multinomial(4);

  // expected uniforms will start from counter offset of 4
  assert_with_expected_uniforms(4);
}

__managed__ int keys[] = {
  1, (1 << 15) + 1,  (1 << 16) + 1,
  2, (1 << 14) + 2, 2
};

__managed__ int values[] = { 1, 2, 3, 4, 5, 9999 };

std::vector<std::vector<int>> valid_perms1 = {
  {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}
};
std::vector<std::vector<int>> valid_perms2 = {
  {4, 5}, {5, 4}
};

TEST(RandomPermutationTest, TestIslandShuffle) {
  if (!at::cuda::is_available()) return;
  at::manual_seed(123);

  bool shuffled1 = false;
  bool shuffled2 = false;
  for (int i = 0; i < 100; i++) {
    cudaDeviceSynchronize();
    std::optional<at::Generator> gen = std::nullopt;
    randperm_handle_duplicate_keys(keys, values, 8, 5, gen);
    cudaDeviceSynchronize();
    std::vector<int> slice1 = {values[0], values[1], values[2]};
    std::vector<int> slice2 = {values[3], values[4]};
    if (slice1 != valid_perms1[0]) {
      shuffled1 = true;
    }
    if (slice2 != valid_perms2[0]) {
      shuffled2 = true;
    }
    bool passed1 = false;
    bool passed2 = false;
    for (auto &i : valid_perms1) {
      if (i == slice1) {
        passed1 = true;
        break;
      }
    }
    for (auto &i : valid_perms2) {
      if (i == slice2) {
        passed2 = true;
        break;
      }
    }
    ASSERT_TRUE(passed1);
    ASSERT_TRUE(passed2);
  }
  ASSERT_TRUE(shuffled1);
  ASSERT_TRUE(shuffled2);
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
- `ATen/ATen.h`
- `ATen/cuda/CUDAContext.h`
- `ATen/native/cuda/Randperm.cuh`
- `cuda.h`
- `cuda_runtime.h`
- `vector`
- `curand.h`
- `curand_kernel.h`
- `curand_philox4x32_x.h`


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
python aten/src/ATen/test/cuda_distributions_test.cu
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
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `cuda_distributions_test.cu_docs.md`
- **Keyword Index**: `cuda_distributions_test.cu_kw.md`
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
python docs/aten/src/ATen/test/cuda_distributions_test.cu_docs.md
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

- **File Documentation**: `cuda_distributions_test.cu_docs.md_docs.md`
- **Keyword Index**: `cuda_distributions_test.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
