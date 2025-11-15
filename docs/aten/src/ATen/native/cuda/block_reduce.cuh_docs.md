# Documentation: block_reduce.cuh

## File Metadata
- **Path**: `aten/src/ATen/native/cuda/block_reduce.cuh`
- **Size**: 4506 bytes
- **Lines**: 149
- **Extension**: .cuh
- **Type**: Regular file

## Original Source

```cuh
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

This file is part of the PyTorch repository. It is a source or configuration file.

## Detailed Walkthrough


## Key Components

The file contains 673 words across 149 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4506 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
