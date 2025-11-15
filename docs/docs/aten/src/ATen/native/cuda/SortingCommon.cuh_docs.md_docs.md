# Documentation: `docs/aten/src/ATen/native/cuda/SortingCommon.cuh_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/SortingCommon.cuh_docs.md`
- **Size**: 8,145 bytes (7.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/SortingCommon.cuh`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/SortingCommon.cuh`
- **Size**: 5,456 bytes (5.33 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

namespace at::native {

// Is this questionable namespace pollution?
#if defined(USE_ROCM)
constexpr int MAX_BLOCK_SIZE = 256;

#else
constexpr int MAX_BLOCK_SIZE = 1024;
#endif

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
constexpr int64_t MAX_GRID_SIZE = 65535LL;

inline bool getGridFromTiles(int64_t gridTiles, dim3& grid) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }

  int64_t gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  int64_t gridY = 1;
  int64_t gridZ = 1;

  if (gridTiles > MAX_GRID_SIZE) {
    gridTiles = ceil_div(gridTiles, MAX_GRID_SIZE);
    gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = ceil_div(gridTiles, MAX_GRID_SIZE);
      gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    }
  }

  grid = dim3(gridX, gridY, gridZ);
  return true;
}

template <typename scalar_t, bool handleNaN = false>
struct GTOp {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && at::_isnan(lhs) && !at::_isnan(rhs)) ||
        (static_cast<scalar_t>(lhs) > static_cast<scalar_t>(rhs));
  }
};

template <typename scalar_t, bool handleNaN = false>
struct LTOp {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && at::_isnan(rhs) && !at::_isnan(lhs)) ||
        (static_cast<scalar_t>(lhs) < static_cast<scalar_t>(rhs));
  }
};

template <typename index_t>
__device__ __forceinline__ index_t getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
      blockIdx.x;
}

// For slice sorting in Thrust; extracts a slice index from a linear
// index and uses that for comparison
struct SliceComp {
  SliceComp(int64_t size) : sliceSize(size) {}

  __device__ bool operator()(const int64_t& a, const int64_t& b) const {
    // Since the slices are guaranteed to be innermost,
    // the segment is just via int64_t division
    int64_t segA = a / sliceSize;
    int64_t segB = b / sliceSize;
    return segA < segB;
  }

  const int64_t sliceSize;
};

// For sorting in Thurst; extracts a within-slice index from a linear index
struct GlobalIndexToPerSliceIndex {
  GlobalIndexToPerSliceIndex(int64_t size) : sliceSize(size) {}

  __device__ inline void operator()(int64_t& v) const {
    v = v % sliceSize;
  }

  const int64_t sliceSize;
};

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
inline uint64_t nextHighestPowerOf2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#ifndef _MSC_VER
  n |= n >> 32;
#endif
  n++;

  return n;
}


// WARNING: This function assumes input tensors are contiguous
template <typename scalar_t, typename index_t, typename Launcher>
void run_launcher(
    const TensorBase &values,
    const TensorBase &indices,
    const TensorBase &self,
    int64_t dim,
    Launcher l) {
  auto self_info = cuda::detail::getTensorInfo<const scalar_t, index_t>(self);
  auto values_info = cuda::detail::getTensorInfo<scalar_t, index_t>(values);
  auto indices_info = cuda::detail::getTensorInfo<int64_t, index_t>(indices);

  int64_t slice_size = self.size(dim);
  /* We use these structures solely to find the offset to */
  /* each slice we are operating on */
  self_info.reduceDim(dim);
  values_info.reduceDim(dim);
  indices_info.reduceDim(dim);

  /* Collapse all other dims */
  int collapse_self_dim = self_info.collapseDims(dim);
  int collapse_values_dim = values_info.collapseDims(dim);
  int collapse_indices_dim = indices_info.collapseDims(dim);

  int64_t num_slices = 1;
  for (int i = 0; i < self_info.dims; ++i) {
    num_slices *= self_info.sizes[i];
  }

  /* This is used as a template parameter to calculate indices. */
  /* We only specialize it if all collapsed dim sizes are the */
  /* same; otherwise, we use -1 which is the specialization */
  /* parameter for arbitrary dimensions */
  int all_dims = self_info.dims;
  if (values_info.dims != all_dims || indices_info.dims != all_dims) {
    all_dims = -1;
  }

  if (all_dims == 1) {
    l.template launch<scalar_t, index_t, 1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 2) {
    l.template launch<scalar_t, index_t, 2>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 3) {
    l.template launch<scalar_t, index_t, 3>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else {
    l.template launch<scalar_t, index_t, -1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  }
}

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/TensorBase.h`
- `ATen/ceil_div.h`
- `ATen/NumericUtils.h`
- `c10/macros/Macros.h`
- `stdlib.h`
- `ATen/cuda/detail/IndexUtils.cuh`
- `ATen/cuda/detail/TensorInfo.cuh`


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

- **File Documentation**: `SortingCommon.cuh_docs.md`
- **Keyword Index**: `SortingCommon.cuh_kw.md`
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

- **File Documentation**: `SortingCommon.cuh_docs.md_docs.md`
- **Keyword Index**: `SortingCommon.cuh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
