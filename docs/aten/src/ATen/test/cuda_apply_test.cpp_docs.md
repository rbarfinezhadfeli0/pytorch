# Documentation: `aten/src/ATen/test/cuda_apply_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/cuda_apply_test.cpp`
- **Size**: 4,785 bytes (4.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAContext.h>
#define ASSERT_EQ_CUDA(X, Y) \
  {                          \
    bool _isEQ = X == Y;     \
    ASSERT_TRUE(_isEQ);      \
  }
/*
   Tests related to tensor indexing and applying operations.
*/
#ifndef _WIN32

// CATCH_TEST_CASE("2D Contiguous", "Collapses a 2D contiguous tensor to 1D
// contiguous") {
TEST(ApplyTest, Contiguous2D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {4, 4};
  int strides[] = {4, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 2, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 1);
  ASSERT_EQ_CUDA(ti.sizes[0], (4 * 4));
}

// CATCH_TEST_CASE("3D Contiguous", "Collapses a 3D contiguous tensor to a 1D
// contiguous") {
TEST(ApplyTest, Contiguous3D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {6, 3, 7};
  int strides[] = {3 * 7, 7, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 1);
  ASSERT_EQ_CUDA(ti.sizes[0], (6 * 3 * 7));
}
// CATCH_TEST_CASE("3D Partial Collapse", "Collapses a 3D noncontiguous tensor
// to a 2D tensor") {
TEST(ApplyTest, PartialCollapse3D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {4, 3, 2};
  int strides[] = {3 * 3, 3, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 2);
  ASSERT_EQ_CUDA(ti.sizes[0], (4 * 3));
  ASSERT_EQ_CUDA(ti.sizes[1], 2);
}

// Collapses a 2D skip contiguous tensor to a 1D skip contiguous tensor
TEST(ApplyTest, StridedCollapse2D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {3, 2};
  int strides[] = {2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 2, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 1);
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 2));
  ASSERT_EQ_CUDA(ti.strides[0], 2);
}

// Collapses a 4D tensor to a 2D tensor
TEST(ApplyTest, PartialStridedCollapse4D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 2);
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 6));
  ASSERT_EQ_CUDA(ti.strides[0], 22);
  ASSERT_EQ_CUDA(ti.sizes[1], (5 * 2));
  ASSERT_EQ_CUDA(ti.strides[1], 2);
}

// Collapses a 5D tensor to a 1D tensor
TEST(ApplyTest, CollapsesZerosAndOnes) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {1, 10, 1, 5, 4};
  int strides[] = {4, 0, 16, 0, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 5, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 2);
  ASSERT_EQ_CUDA(ti.sizes[0], (10 * 5));
  ASSERT_EQ_CUDA(ti.strides[0], 0);
  ASSERT_EQ_CUDA(ti.sizes[1], 4);
  ASSERT_EQ_CUDA(ti.strides[1], 1);
}

// Collapses a 3D tensor to a point tensor
TEST(ApplyTest, CollapseToPointTensor) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {1, 1, 1};
  int strides[] = {17, 12, 3};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ASSERT_EQ_CUDA(ti.collapseDims(), 0);
  ASSERT_EQ_CUDA(ti.dims, 1);
  ASSERT_EQ_CUDA(ti.sizes[0], 1);
  ASSERT_EQ_CUDA(ti.strides[0], 1);
}

// Collapses a 4D tensor to a 3D tensor
TEST(ApplyTest, ExcludingInContiguous4D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  ASSERT_EQ_CUDA(ti.collapseDims(1), 1);
  ASSERT_EQ_CUDA(ti.dims, 3);
  ASSERT_EQ_CUDA(ti.sizes[0], 3);
  ASSERT_EQ_CUDA(ti.strides[0], (6 * 22));
  ASSERT_EQ_CUDA(ti.sizes[1], 6);
  ASSERT_EQ_CUDA(ti.strides[1], 22);
  ASSERT_EQ_CUDA(ti.sizes[2], (5 * 2));
  ASSERT_EQ_CUDA(ti.strides[2], 2);
}

// Collapses a 4D tensor to a 3D tensor
TEST(ApplyTest, RovingExclusion) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  ASSERT_EQ_CUDA(ti.collapseDims(2), 1);
  ASSERT_EQ_CUDA(ti.dims, 3);
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 6));
  ASSERT_EQ_CUDA(ti.strides[0], 22);
  ASSERT_EQ_CUDA(ti.sizes[1], 5);
  ASSERT_EQ_CUDA(ti.strides[1], 4);
  ASSERT_EQ_CUDA(ti.sizes[2], 2);
  ASSERT_EQ_CUDA(ti.strides[2], 2);
}

// Attempts to exclude a nonexisting dimension
TEST(ApplyTest, InvalidExclusion) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {1, 1, 1};
  int strides[] = {17, 12, 3};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ASSERT_ANY_THROW(ti.collapseDims(5));
}
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

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
- `cuda.h`
- `cuda_runtime.h`
- `ATen/cuda/detail/TensorInfo.cuh`
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
python aten/src/ATen/test/cuda_apply_test.cpp
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

- **File Documentation**: `cuda_apply_test.cpp_docs.md`
- **Keyword Index**: `cuda_apply_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
