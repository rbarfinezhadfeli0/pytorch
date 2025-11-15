# Documentation: `docs/aten/src/ATen/native/cuda/ForeachFunctors.cuh_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ForeachFunctors.cuh_docs.md`
- **Size**: 27,329 bytes (26.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ForeachFunctors.cuh`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ForeachFunctors.cuh`
- **Size**: 24,686 bytes (24.11 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/Pow.cuh>

namespace at::native {

namespace {

// TODO(crcrpar): Handle version bump in codegen.
// rel:
// https://github.com/pytorch/pytorch/blob/9cf84347767c8abb8feba18a9a1baba321eeb8b9/tools/autograd/gen_inplace_or_view_type.py#L481-L482
inline void increment_version(TensorList tensors) {
  for (const auto& t : tensors) {
    t.unsafeGetTensorImpl()->bump_version();
  }
}

// Initializes args and checks if all args are aligned
template <int depth, typename T>
__device__ bool init_args(
    T** args,
    TensorListMetadata<depth>& tl,
    const int64_t chunk_idx,
    const int64_t chunk_size,
    const int64_t tensor_loc) {
  bool all_aligned = true;
  for (int i = 0; i < depth; i++) {
    args[i] = (T*)tl.addresses[i][tensor_loc];
    args[i] += chunk_idx * chunk_size;

    if (!is_aligned(args[i])) {
      all_aligned = false;
    }
  }
  return all_aligned;
}

// Initializes args and checks if all args are aligned
template <int depth, typename T, typename T2>
__device__ bool init_args(
    T** args,
    TensorListScalarListMetadata<T2, depth>& tl,
    const int64_t chunk_idx,
    const int64_t chunk_size,
    const int64_t tensor_loc) {
  bool all_aligned = true;
  for (int i = 0; i < depth; i++) {
    args[i] = (T*)tl.addresses[i][tensor_loc];
    args[i] += chunk_idx * chunk_size;

    if (!is_aligned(args[i])) {
      all_aligned = false;
    }
  }
  return all_aligned;
}

template <int depth, typename T>
__device__ bool init_args(
    T** args,
    FusedOptimizerTensorListMetadata<depth>& tl,
    const int64_t chunk_idx,
    const int64_t chunk_size,
    const int64_t tensor_loc) {
  bool all_aligned = true;
  for (int i = 0; i < depth; i++) {
    args[i] = (T*)tl.addresses[i][tensor_loc];
    args[i] += chunk_idx * chunk_size;

    if (!is_aligned(args[i])) {
      all_aligned = false;
    }
  }
  return all_aligned;
}

template <int depth, typename T>
__device__ void load_args(
    T r_args[][kILP],
    T** args,
    const int64_t i_start,
    const int64_t chunk_size,
    const int64_t n) {
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    const auto i = i_start + threadIdx.x + ii * blockDim.x;
    for (int r_index = 0; r_index < depth; r_index++) {
      r_args[r_index][ii] = 0;
      if (i < n && i < chunk_size) {
        r_args[r_index][ii] = args[r_index][i];
      }
    }
  }
}

template <typename T>
__device__ void store_args(
    T* dst,
    T* src,
    const int64_t i_start,
    const int64_t chunk_size,
    const int64_t n) {
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    const int64_t i = i_start + threadIdx.x + ii * blockDim.x;
    if (i < n && i < chunk_size)
      dst[i] = src[ii];
  }
}

template <int res_arg_index, typename Op, typename T, typename opmath_t>
__device__ __forceinline__ void binary_op_scalar(
    T r_args[][kILP],
    T** args,
    opmath_t scalar,
    const int64_t n,
    const int64_t chunk_size,
    const bool all_aligned,
    Op op) {
  // to make things simple, we put aligned case in a different code path
  if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
    for (int64_t i_start = threadIdx.x;
         i_start * kILP < n && i_start * kILP < chunk_size;
         i_start += blockDim.x) {
      // load
      load_store(r_args[0], args[0], 0, i_start);
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        r_args[0][ii] = static_cast<T>(
            op(static_cast<opmath_t>(r_args[0][ii]),
               static_cast<opmath_t>(scalar)));
      }
      // store
      load_store(args[res_arg_index], r_args[0], i_start, 0);
    }
  } else {
    for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
         i_start += blockDim.x * kILP) {
      // Regardless if depth is 1 (for inplace) or 2 (for out of place), r_args
      // has depth 1
      load_args<1>(r_args, args, i_start, chunk_size, n);
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        r_args[0][ii] = static_cast<T>(
            op(static_cast<opmath_t>(r_args[0][ii]),
               static_cast<opmath_t>(scalar)));
      }
      store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
    }
  }
}

template <int res_arg_index, typename Op, typename T, typename opmath_t>
__device__ __forceinline__ void pointwise_op_scalar(
    T r_args[][kILP],
    T** args,
    opmath_t scalar,
    const int64_t n,
    const int64_t chunk_size,
    const bool all_aligned,
    Op op) {
  // to make things simple, we put aligned case in a different code path
  if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
    for (int64_t i_start = threadIdx.x;
         i_start * kILP < n && i_start * kILP < chunk_size;
         i_start += blockDim.x) {
      // load
      load_store(r_args[0], args[0], 0, i_start);
      load_store(r_args[1], args[1], 0, i_start);
      load_store(r_args[2], args[2], 0, i_start);
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        r_args[0][ii] = static_cast<T>(
            static_cast<opmath_t>(r_args[0][ii]) +
            scalar *
                op(static_cast<opmath_t>(r_args[1][ii]),
                   static_cast<opmath_t>(r_args[2][ii])));
      }
      // store
      load_store(args[res_arg_index], r_args[0], i_start, 0);
    }
  } else {
    for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
         i_start += blockDim.x * kILP) {
      // Regardless if depth is 3 (for inplace) or 4 (for out of place), r_args
      // has depth 3
      load_args<3>(r_args, args, i_start, chunk_size, n);
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        r_args[0][ii] = static_cast<T>(
            static_cast<opmath_t>(r_args[0][ii]) +
            scalar *
                op(static_cast<opmath_t>(r_args[1][ii]),
                   static_cast<opmath_t>(r_args[2][ii])));
      }
      store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
    }
  }
}

//
// Binary Functors
//
template <typename T, int depth, int r_args_depth, int res_arg_index>
struct BinaryOpScalarFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      Op op,
      opmath_t scalar) {
    const int tensor_loc = tl.block_to_tensor[blockIdx.x];
    const int chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    binary_op_scalar<res_arg_index>(
        r_args, args, scalar, n, chunk_size, all_aligned, op);
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct BinaryOpScalarListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListScalarListMetadata<opmath_t, depth>& tl,
      Op op) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    opmath_t scalar = tl.scalar_vals[tensor_loc];
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    binary_op_scalar<res_arg_index>(
        r_args, args, scalar, n, chunk_size, all_aligned, op);
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct BinaryOpListAlphaFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      Op op,
      opmath_t alpha) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(
              op(static_cast<opmath_t>(r_args[0][ii]),
                 alpha * static_cast<opmath_t>(r_args[1][ii])));
        }
        // store
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<r_args_depth>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(
              op(static_cast<opmath_t>(r_args[0][ii]),
                 alpha * static_cast<opmath_t>(r_args[1][ii])));
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct BinaryOpScalarTensorFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      Op op,
      T* scalar,
      opmath_t alpha) {
    const int tensor_loc = tl.block_to_tensor[blockIdx.x];
    const int chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(op(
              static_cast<opmath_t>(r_args[0][ii]),
              static_cast<opmath_t>(alpha) * static_cast<opmath_t>(*scalar)));
        }
        // store
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        // Regardless if depth is 1 (for inplace) or 2 (for out of place),
        // r_args has depth 1
        load_args<1>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(op(
              static_cast<opmath_t>(r_args[0][ii]),
              static_cast<opmath_t>(alpha) * static_cast<opmath_t>(*scalar)));
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

//
// Unary Functors
//

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct ZeroFunctor {
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<1>& tl) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const auto all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = 0;
        }
        // store
        load_store(args[0], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = 0;
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct UnaryOpFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      Op op) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              static_cast<T>(op(static_cast<opmath_t>(r_args[0][ii])));
        }
        // store
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<r_args_depth>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              static_cast<T>(op(static_cast<opmath_t>(r_args[0][ii])));
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

//
// Pointwise Functors
//

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct PointwiseOpScalarFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      Op op,
      opmath_t scalar) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    pointwise_op_scalar<res_arg_index>(
        r_args, args, scalar, n, chunk_size, all_aligned, op);
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct PointwiseOpScalarListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListScalarListMetadata<opmath_t, depth>& tl,
      Op op) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    opmath_t scalar = tl.scalar_vals[tensor_loc];
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    pointwise_op_scalar<res_arg_index>(
        r_args, args, scalar, n, chunk_size, all_aligned, op);
  }
};

template <typename T, int depth>
struct PointwiseOpListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      Op op) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[depth - 1][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii])));
        }
        // store
        load_store(args[2], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<depth - 1>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii])));
        }
        store_args(args[2], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct TernaryOpListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      Op op) {
    static_assert(depth == 3 || depth == 4, "");
    static_assert(depth >= r_args_depth, "");
    static_assert(res_arg_index == depth - 1 || res_arg_index == 0, "");
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
        load_store(r_args[2], args[2], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 static_cast<opmath_t>(r_args[2][ii]));
        }
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<r_args_depth>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 static_cast<opmath_t>(r_args[2][ii]));
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct TernaryOpScalarFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      Op op,
      opmath_t alpha) {
    static_assert(depth == 2 || depth == 3, "");
    static_assert(depth >= r_args_depth, "");
    static_assert(res_arg_index == depth - 1 || res_arg_index == 0, "");
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 alpha);
        }
        // store
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<r_args_depth>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 alpha);
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct TernaryOpScalarListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListScalarListMetadata<opmath_t, depth>& tl,
      Op op) {
    static_assert(depth == 2 || depth == 3, "");
    static_assert(depth >= r_args_depth, "");
    static_assert(res_arg_index == depth - 1 || res_arg_index == 0, "");
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];
    const opmath_t scalar = tl.scalar_vals[tensor_loc];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 scalar);
        }
        // store
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<r_args_depth>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 scalar);
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

template <typename T>
struct power_functor {
  C10_DEVICE T operator()(const T& a, const T& b) const {
    return at::native::pow_(a, b);
  }
};

template <typename T>
struct reverse_power_functor {
  C10_DEVICE T operator()(const T& a, const T& b) const {
    return at::native::pow_(b, a);
  }
};

} // namespace
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

- `ATen/OpMathType.h`
- `ATen/native/ForeachUtils.h`
- `ATen/native/cuda/MultiTensorApply.cuh`
- `ATen/native/cuda/Pow.cuh`


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

- **File Documentation**: `ForeachFunctors.cuh_docs.md`
- **Keyword Index**: `ForeachFunctors.cuh_kw.md`
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

- **File Documentation**: `ForeachFunctors.cuh_docs.md_docs.md`
- **Keyword Index**: `ForeachFunctors.cuh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
