# Documentation: `aten/src/ATen/test/cuda_allocator_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/cuda_allocator_test.cpp`
- **Size**: 2,935 bytes (2.87 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/test/allocator_clone_test.h>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

std::unordered_map<void*, size_t> allocation_sizes;

void* logging_malloc(size_t size, int device, cudaStream_t stream) {
    void* ptr;
    cudaMalloc(&ptr, size);
    allocation_sizes[ptr] = size;
    return ptr;
}

void logging_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (allocation_sizes.find(ptr) != allocation_sizes.end()) {
        if (allocation_sizes[ptr] != size) {
          throw std::runtime_error("free mismatch");
        }
    } else {
      throw std::runtime_error("free of unknown ptr");
    }
    cudaFree(ptr);
    allocation_sizes.erase(ptr);
}

TEST(TestTorchUnique, UniqueComparisonTest) {
  if (!at::cuda::is_available()) return;
  auto custom_allocator =
      torch::cuda::CUDAPluggableAllocator::createCustomAllocator(logging_malloc, logging_free);
  torch::cuda::CUDAPluggableAllocator::changeCurrentAllocator(custom_allocator);
  // Run the command 3 times; the first 2 will pass and the third invocation will have
  // different sizes in alloc and free if the test fails.
  for (int i = 0; i < 3; ++i) {
    // Initialize simple sorted tensor with repeats
    at::Tensor sorted_tensor =
        at::tensor({0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 5},
                      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

    // This operation will call malloc/free with different sizes on the same pointer
    auto unique_dim_result = at::unique_consecutive(sorted_tensor, false, true, 0);

    // Everything below is only there to validate correct results
    auto unique_dim_values = std::get<0>(unique_dim_result);
    auto unique_dim_counts = std::get<2>(unique_dim_result);

    // Check tensor sizes
    EXPECT_EQ(unique_dim_values.size(0), 5);
    EXPECT_EQ(unique_dim_counts.size(0), 5);

    // Copy to CPU before accessing elements
    at::Tensor cpu_values = unique_dim_values.cpu();
    at::Tensor cpu_counts = unique_dim_counts.cpu();

    // Use accessors on the CPU tensors
    auto values_accessor = cpu_values.accessor<float, 1>();
    auto counts_accessor = cpu_counts.accessor<int64_t, 1>();

    // Check individual values using accessors
    EXPECT_EQ(values_accessor[0], 0.0f);
    EXPECT_EQ(values_accessor[1], 1.0f);
    EXPECT_EQ(values_accessor[2], 2.0f);
    EXPECT_EQ(values_accessor[3], 3.0f);
    EXPECT_EQ(values_accessor[4], 5.0f);

    // Check count values using accessors
    EXPECT_EQ(counts_accessor[0], 3);
    EXPECT_EQ(counts_accessor[1], 2);
    EXPECT_EQ(counts_accessor[2], 1);
    EXPECT_EQ(counts_accessor[3], 4);
    EXPECT_EQ(counts_accessor[4], 1);
  }
}

TEST(AllocatorTestCUDA, test_clone) {
  if (!at::cuda::is_available()) return;
  test_allocator_clone(c10::cuda::CUDACachingAllocator::get());
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

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
- `c10/cuda/CUDACachingAllocator.h`
- `ATen/test/allocator_clone_test.h`
- `torch/csrc/cuda/CUDAPluggableAllocator.h`


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
python aten/src/ATen/test/cuda_allocator_test.cpp
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

- **File Documentation**: `cuda_allocator_test.cpp_docs.md`
- **Keyword Index**: `cuda_allocator_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
