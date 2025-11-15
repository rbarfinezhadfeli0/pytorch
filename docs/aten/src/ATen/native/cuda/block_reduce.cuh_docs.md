# Documentation: `aten/src/ATen/native/cuda/block_reduce.cuh`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/block_reduce.cuh`
- **Size**: 4,506 bytes (4.40 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once

#include <thrust/tuple.h>

#include <ATen/native/SharedReduceOps.h>
#include <ATen/cuda/DeviceUtils.cuh>

namespace at::native::cuda_utils {

constexpr int kCUDABlockReduceNumThreads = 512;
// Algorithmic limitation: BlockReduce does two WarpReduce calls, each
// of which reduces C10_WARP_SIZE elements. So, at most
// C10_WARP_SIZE**2 elements can be reduced at a time.
// NOTE: This is >= the max block size on current hardware anyway (1024).
// ROCm NOTE: C10_WARP_SIZE should only be used inside device functions,
// and kCUDABlockReduceMaxThreads is a host-side variable.
#ifdef USE_ROCM
static int kCUDABlockReduceMaxThreads() {
    return at::cuda::warp_size() * at::cuda::warp_size();
}
#else
constexpr int kCUDABlockReduceMaxThreads() {
    return C10_WARP_SIZE * C10_WARP_SIZE;
}
#endif

// Sums `val` across all threads in a warp.
//
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

// Picks the maximum `val` across all threads in a warp.
//
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
template <typename T>
__inline__ __device__ T WarpReduceMax(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = max_propagate_nan(val, WARP_SHFL_DOWN(val, offset));
  }
  return val;
}

struct Block1D {
    static __forceinline__ __device__ int Tid() { return threadIdx.x; }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x / C10_WARP_SIZE;
    }
};

struct Block2D {
    static __forceinline__ __device__ int Tid() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x * blockDim.y / C10_WARP_SIZE;
    }
};

// Sums `val` across all threads in a block.
//
// Warning: the return value is only valid for thread 0.
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

// Picks out the maximum `val` across all threads in a block.
//
// Warning: the return value is only valid for thread 0.
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceMax(T val, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduceMax(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : T(std::numeric_limits<T>::lowest());
  if (wid == 0) {
    val = WarpReduceMax(val);
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp, typename B = Block1D>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}

} // namespace at::native::cuda_utils

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

- `thrust/tuple.h`
- `ATen/native/SharedReduceOps.h`
- `ATen/cuda/DeviceUtils.cuh`


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

- **File Documentation**: `block_reduce.cuh_docs.md`
- **Keyword Index**: `block_reduce.cuh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
