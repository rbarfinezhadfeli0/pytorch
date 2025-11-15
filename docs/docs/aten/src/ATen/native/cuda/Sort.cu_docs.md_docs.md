# Documentation: `docs/aten/src/ATen/native/cuda/Sort.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/Sort.cu_docs.md`
- **Size**: 15,384 bytes (15.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/Sort.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Sort.cu`
- **Size**: 12,492 bytes (12.20 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/Sort.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/native/cuda/SortUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>

#include <limits>

namespace at::native {

template <typename T>
static int minimum_grid_for_occupancy(T kernel, int max_block_size) {
  int minGridSize = 0;
  int blockSize = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &minGridSize,
      &blockSize,
      kernel,
      /*dynamicSMemSize=*/0,
      max_block_size));
  return minGridSize;
}

template <typename T>
constexpr bool has_nan() {
  if constexpr (std::numeric_limits<T>::is_specialized) {
    return std::numeric_limits<T>::has_quiet_NaN;
  } else if constexpr (
      c10::is_complex<T>::value ||
      std::is_same_v<T, c10::BFloat16> ||
      std::is_same_v<T, c10::Half>) {
    return true;
  }
}

// For very small unstable sorts (n <= 32), use bitonicSortKVInPlace
// which can sort multiple arrays within the same block of threads,
// improving occupancy.
struct SmallBitonicSort {
  template <int A, typename K, typename V, typename IndexType>
  void sort(
      at::cuda::detail::TensorInfo<K, IndexType> keyInfo,
      IndexType keySlices,
      IndexType keySliceSize,
      IndexType keySliceStride,
      at::cuda::detail::TensorInfo<V, IndexType> valueInfo,
      IndexType valueSliceStride,
      bool descending) {
    constexpr int sort_size = 32;
    constexpr int max_block_y = 16;
    constexpr int items_per_thread = 2;
    static_assert(sort_size % items_per_thread == 0, "");
    constexpr int block_x = sort_size / items_per_thread;

    TORCH_INTERNAL_ASSERT(keySliceSize <= sort_size);

    // Scale batch size down if the grid would be too small
    const auto min_grid = minimum_grid_for_occupancy(
        bitonicSortKVInPlace<
            A, -1, block_x, max_block_y,
            K, V, LTOp<K, true>, IndexType>,
        block_x * max_block_y);
    const auto max_batch = std::max(IndexType{1}, keySlices / min_grid);
    const int block_y = std::min(IndexType(max_block_y), max_batch);
    dim3 block(block_x, block_y);

    dim3 grid;
    const int grid_count = (keySlices + block_y - 1) / block_y;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(grid_count, grid),
                          "Too many slices to sort");
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (descending) {
      bitonicSortKVInPlace<A, -1, block_x, max_block_y>
        <<<grid, block, 0, stream>>>(
          keyInfo,
          keySlices,
          keySliceSize,
          keySliceStride,
          valueInfo,
          valueSliceStride,
          GTOp<K, true>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      bitonicSortKVInPlace<A, -1, block_x, max_block_y>
        <<<grid, block, 0, stream>>>(
          keyInfo,
          keySlices,
          keySliceSize,
          keySliceStride,
          valueInfo,
          valueSliceStride,
          LTOp<K, true>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
};

#if HAS_WARP_MERGE_SORT()

// For small sorts (n <= 128) we use warpMergeSortKVInPlace which
// sorts one slice per warp and potentially multiple slices in the
// same block for improved occupancy with large batch sizes.
template <int sort_size>
struct WarpMergeSort {

  template <int A, typename K, typename V, typename IndexType>
  void sort(
      at::cuda::detail::TensorInfo<K, IndexType> keyInfo,
      IndexType keySlices,
      IndexType keySliceSize,
      IndexType keySliceStride,
      at::cuda::detail::TensorInfo<V, IndexType> valueInfo,
      IndexType valueSliceStride,
      bool descending) {
    constexpr int max_block_y = 16;
    const int block_x = at::cuda::warp_size();

    TORCH_INTERNAL_ASSERT(keySliceSize <= sort_size);

    // Scale batch size down if the grid would be too small
    const auto min_grid = minimum_grid_for_occupancy(
        warpMergeSortKVInPlace<
            A, -1, sort_size, max_block_y,
            K, V, LTOp<K, true>, IndexType>,
        block_x * max_block_y);
    const auto max_batch = std::max(IndexType{1}, keySlices / min_grid);
    const int block_y = std::min(IndexType(max_block_y), max_batch);
    dim3 block(block_x, block_y);

    dim3 grid;
    const int grid_count = (keySlices + block_y - 1) / block_y;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(grid_count, grid),
                          "Too many slices to sort");
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (descending) {
      const K invalid_key = at::numeric_limits<K>::lower_bound();
      warpMergeSortKVInPlace<A, -1, sort_size, max_block_y>
        <<<grid, block, 0, stream>>>(
          keyInfo,
          keySlices,
          keySliceSize,
          keySliceStride,
          valueInfo,
          valueSliceStride,
          GTOp<K, true>(),
          invalid_key);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      const K invalid_key = []{
        // NAN is sorted after inf
        if constexpr(has_nan<K>()) {
          return K(NAN);
        }
        return at::numeric_limits<K>::upper_bound();
      }();
      warpMergeSortKVInPlace<A, -1, sort_size, max_block_y>
        <<<grid, block, 0, stream>>>(
          keyInfo,
          keySlices,
          keySliceSize,
          keySliceStride,
          valueInfo,
          valueSliceStride,
          LTOp<K, true>(),
          invalid_key);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
};

#endif // !HAS_WARP_MERGE_SORT()

// For medium sizes (128 < n <= 4096) use radixSortKVInplace.
struct MediumRadixSort {

  template <int A, typename K, typename V, typename IndexType>
  void sort(
      at::cuda::detail::TensorInfo<K, IndexType> keyInfo,
      IndexType keySlices,
      IndexType keySliceSize,
      IndexType keySliceStride,
      at::cuda::detail::TensorInfo<V, IndexType> valueInfo,
      IndexType valueSliceStride,
      bool descending) {

#define HANDLE_CASE(SIZE, ITEMS_PER_THREAD)         \
    fixed_size_sort<A, SIZE, ITEMS_PER_THREAD>(     \
        keyInfo,                                    \
        keySlices,                                  \
        keySliceSize,                               \
        keySliceStride,                             \
        valueInfo,                                  \
        valueSliceStride,                           \
        descending)

    int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);
    TORCH_INTERNAL_ASSERT(ceilPowerOf2 <= 4096);
#ifdef USE_ROCM
    constexpr int default_ipt = 8;
#else
    constexpr int default_ipt = 32;
#endif
    switch (ceilPowerOf2) {
      case 4096:
        HANDLE_CASE(4096, default_ipt);
        break;
      case 2048:
        HANDLE_CASE(2048, default_ipt);
        break;
      case 1024:
      case 512:
      case 256:
        HANDLE_CASE(1024, default_ipt);
        break;
      case 128:
      case 64:
#if !HAS_WARP_MERGE_SORT()
        HANDLE_CASE(128, 4);
        break;
#endif
      case 32:
      case 16:
      case 8:
      case 4:
      case 2:
#if HAS_WARP_MERGE_SORT()
        TORCH_INTERNAL_ASSERT(
            false, "Expected size <= 128 to be handled by a different algorithm");
#else
        HANDLE_CASE(32, 2);
#endif
        break;
      case 1:
        /* Nothing to do, data already sorted */
        break;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
#undef HANDLE_CASE

  }

  template <int A, int sort_size, int items_per_thread,
            typename K, typename V, typename IndexType>
  void fixed_size_sort(
      at::cuda::detail::TensorInfo<K, IndexType> keyInfo,
      IndexType keySlices,
      IndexType keySliceSize,
      IndexType keySliceStride,
      at::cuda::detail::TensorInfo<V, IndexType> valueInfo,
      IndexType valueSliceStride,
      bool descending) {
    static_assert(sort_size % items_per_thread == 0, "");
    constexpr int block = sort_size / items_per_thread;
    dim3 grid;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(keySlices, grid),
                          "Too many slices to sort");

    const auto stream = at::cuda::getCurrentCUDAStream();
    radixSortKVInPlace<A, -1, block, items_per_thread>
        <<<grid, block, 0, stream>>>(
          keyInfo,
          keySlices,
          keySliceSize,
          keySliceStride,
          valueInfo,
          valueSliceStride,
          descending);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};

template <typename Sorter>
void sortCommon(Sorter sorter, const TensorBase &key, const TensorBase &value,
                int dim, bool descending) {
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key tensor must have same size as value tensor");
  int dims = value.dim();
  TORCH_CHECK(dims <= MAX_DIMS, "value tensor has too many dimensions");
  // if key and value tensors have the same size, we do not need to check both

  ptrdiff_t inElements = key.numel();

  if (inElements == 0) {
    return;
  }

  int64_t keySliceSize = key.size(dim);
  ptrdiff_t keySlices = inElements / keySliceSize;

#define HANDLE_SORT_CASE(TYPE, A)                   \
  sorter.template sort<A>(                          \
      keyInfo,                                      \
      (TYPE) keySlices,                             \
      (TYPE) keySliceSize,                          \
      (TYPE) keyInfo.strides[collapseKeyDim],       \
      valueInfo,                                    \
      (TYPE) valueInfo.strides[collapseValueDim],   \
      descending)

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, key.scalar_type(), "sortKeyValueInplace", [&]  {
    if (at::cuda::detail::canUse32BitIndexMath(key)) {
      at::cuda::detail::TensorInfo<scalar_t, unsigned int> keyInfo =
        at::cuda::detail::getTensorInfo<scalar_t, unsigned int>(key);
      at::cuda::detail::TensorInfo<int64_t, unsigned int> valueInfo =
        at::cuda::detail::getTensorInfo<int64_t, unsigned int>(value);

      auto strideKey = keyInfo.strides[dim];
      keyInfo.sizes[dim] = 1;
      int collapseKeyDim = keyInfo.collapseDims(dim);
      keyInfo.strides[collapseKeyDim] = strideKey;
      auto strideValue = valueInfo.strides[dim];
      valueInfo.sizes[dim]=1;
      int collapseValueDim = valueInfo.collapseDims(dim);
      valueInfo.strides[collapseValueDim] = strideValue;

      if (keyInfo.isContiguous()) {
        HANDLE_SORT_CASE(unsigned int, -2);
      } else {
        switch (keyInfo.dims) {
          case 2:
            HANDLE_SORT_CASE(unsigned int, 2);
            break;
          default:
            HANDLE_SORT_CASE(unsigned int, -1);
            break;
        }
      }

    } else {
      at::cuda::detail::TensorInfo<scalar_t, uint64_t> keyInfo =
        at::cuda::detail::getTensorInfo<scalar_t, uint64_t>(key);
      at::cuda::detail::TensorInfo<int64_t, uint64_t> valueInfo =
        at::cuda::detail::getTensorInfo<int64_t, uint64_t>(value);

      auto strideKey = keyInfo.strides[dim];
      keyInfo.sizes[dim] = 1;
      int collapseKeyDim = keyInfo.collapseDims(dim);
      keyInfo.strides[collapseKeyDim] = strideKey;
      auto strideValue = valueInfo.strides[dim];
      valueInfo.sizes[dim]=1;
      int collapseValueDim = valueInfo.collapseDims(dim);
      valueInfo.strides[collapseValueDim] = strideValue;

      // int64_t case is rare, just instantiate the generic version
      HANDLE_SORT_CASE(uint64_t, -1);
    }
  });
#undef HANDLE_SORT_CASE
}

void sortKeyValueInplace(
    const TensorBase& key,
    const TensorBase& value,
    int64_t dim,
    bool descending,
    bool stable) {
  const auto sort_size = key.size(dim);
  if (sort_size <= 1) {
    return; // Already sorted
  } else if (!stable && sort_size <= 32) {
    // NOTE: Bitonic sort is unstable
    sortCommon(SmallBitonicSort{}, key, value, dim, descending);
#if HAS_WARP_MERGE_SORT()
  } else if (sort_size <= 128) {
    sortCommon(WarpMergeSort<128>{}, key, value, dim, descending);
#endif
  } else {
    sortCommon(MediumRadixSort{}, key, value, dim, descending);
  }
}

}  // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `SmallBitonicSort`, `WarpMergeSort`, `MediumRadixSort`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/cuda/Sort.h`
- `ATen/core/TensorBase.h`
- `ATen/Dispatch.h`
- `ATen/cuda/cub.cuh`
- `ATen/cuda/CUDAContext.h`
- `ATen/cuda/detail/KernelUtils.h`
- `ATen/cuda/detail/OffsetCalculator.cuh`
- `ATen/cuda/NumericLimits.cuh`
- `ATen/native/cuda/SortUtils.cuh`
- `ATen/native/cuda/SortingCommon.cuh`
- `limits`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/cuda`):

- [`LogcumsumexpKernel.cu_docs.md`](./LogcumsumexpKernel.cu_docs.md)
- [`WeightNorm.cu_docs.md`](./WeightNorm.cu_docs.md)
- [`SparseBinaryOpIntersectionKernel.cu_docs.md`](./SparseBinaryOpIntersectionKernel.cu_docs.md)
- [`jit_utils.cpp_docs.md`](./jit_utils.cpp_docs.md)
- [`ReduceNormKernel.cu_docs.md`](./ReduceNormKernel.cu_docs.md)
- [`BinaryMiscOpsKernels.cu_docs.md`](./BinaryMiscOpsKernels.cu_docs.md)
- [`RowwiseScaledMM.h_docs.md`](./RowwiseScaledMM.h_docs.md)
- [`fused_adamw_amsgrad_impl.cuh_docs.md`](./fused_adamw_amsgrad_impl.cuh_docs.md)
- [`Col2Im.cu_docs.md`](./Col2Im.cu_docs.md)
- [`DistributionRandomKernel.cu_docs.md`](./DistributionRandomKernel.cu_docs.md)


## Cross-References

- **File Documentation**: `Sort.cu_docs.md`
- **Keyword Index**: `Sort.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/native/cuda`):

- [`DeviceSqrt.cuh_kw.md_docs.md`](./DeviceSqrt.cuh_kw.md_docs.md)
- [`UnaryGeometricAsinKernel.cu_kw.md_docs.md`](./UnaryGeometricAsinKernel.cu_kw.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`fused_adamw_impl.cu_docs.md_docs.md`](./fused_adamw_impl.cu_docs.md_docs.md)
- [`TensorTopK.h_kw.md_docs.md`](./TensorTopK.h_kw.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`FusedSgdKernel.cu_docs.md_docs.md`](./FusedSgdKernel.cu_docs.md_docs.md)
- [`Distributions.cu_kw.md_docs.md`](./Distributions.cu_kw.md_docs.md)
- [`block_reduce.cuh_docs.md_docs.md`](./block_reduce.cuh_docs.md_docs.md)
- [`fused_adagrad_impl.cuh_kw.md_docs.md`](./fused_adagrad_impl.cuh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Sort.cu_docs.md_docs.md`
- **Keyword Index**: `Sort.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
