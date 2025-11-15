# Documentation: `docs/aten/src/ATen/test/cuda_dlconvertor_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/cuda_dlconvertor_test.cpp_docs.md`
- **Size**: 4,949 bytes (4.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/cuda_dlconvertor_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/cuda_dlconvertor_test.cpp`
- **Size**: 2,212 bytes (2.16 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <cuda.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAContext.h>

using namespace at;

TEST(TestDlconvertor, TestDlconvertorCUDA) {
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensor* dlMTensor = toDLPack(a);

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorNoStridesCUDA) {
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensor* dlMTensor = toDLPack(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorCUDAHIP) {
  if (!at::cuda::is_available())
    return;
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensor* dlMTensor = toDLPack(a);

#if AT_ROCM_ENABLED()
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLROCM);
#else
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLCUDA);
#endif

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorVersioned, TestDlconvertorCUDA) {
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorVersioned, TestDlconvertorNoStridesCUDA) {
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorVersioned, TestDlconvertorCUDAHIP) {
  if (!at::cuda::is_available())
    return;
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);

#if AT_ROCM_ENABLED()
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLROCM);
#else
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLCUDA);
#endif

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cuda.h`
- `cuda_runtime.h`
- `gtest/gtest.h`
- `ATen/ATen.h`
- `ATen/DLConvertor.h`
- `ATen/cuda/CUDAConfig.h`
- `ATen/cuda/CUDAContext.h`


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
python aten/src/ATen/test/cuda_dlconvertor_test.cpp
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

- **File Documentation**: `cuda_dlconvertor_test.cpp_docs.md`
- **Keyword Index**: `cuda_dlconvertor_test.cpp_kw.md`
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
python docs/aten/src/ATen/test/cuda_dlconvertor_test.cpp_docs.md
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

- **File Documentation**: `cuda_dlconvertor_test.cpp_docs.md_docs.md`
- **Keyword Index**: `cuda_dlconvertor_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
