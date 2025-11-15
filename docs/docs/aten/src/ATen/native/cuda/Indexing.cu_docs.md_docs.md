# Documentation: `docs/aten/src/ATen/native/cuda/Indexing.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/Indexing.cu_docs.md`
- **Size**: 53,587 bytes (52.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/Indexing.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Indexing.cu`
- **Size**: 83,675 bytes (81.71 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/Resize.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/DeviceUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_assert_async.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_reduce_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/cub.h>
#include <c10/util/irange.h>
#include <c10/core/QScheme.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>

#include <limits>

#include <c10/macros/Macros.h>

namespace {
constexpr uint64_t getDefaultMaxThreadsPerBlock() {
#ifndef USE_ROCM
  return 128;
#else
  // bigger default
  return 512;
#endif
}

#ifdef USE_ROCM
#define SKIP_SORTED_INDICES 32
template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel_many_indices(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  extern __shared__ unsigned char smem[];
  auto smem_dups_cache = reinterpret_cast<int64_t*>(smem);

  int smem_offset = threadIdx.y * C10_WARP_SIZE;

  int laneIdx = threadIdx.x % C10_WARP_SIZE;
  int64_t grad_row = 0;

  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z) {
    // Init duplicates every time we compute a new set of entries:
    smem_dups_cache[smem_offset + laneIdx] = 0;
    WARP_SYNC();

    int64_t base_idx = blockIdx.x * blockDim.y * C10_WARP_SIZE + threadIdx.y * C10_WARP_SIZE;
    int64_t idx = base_idx + laneIdx;

    if (idx < numel) {
      int64_t crnt_sorted_idx = sorted_indices[idx];

      if (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]) {
        // Determine the number of duplicates in advance:
        int64_t num_duplicates = 1;

        // Lookahead in case there is a large number of duplicates. Once that is done, handle the tail.
        while ((idx + num_duplicates + SKIP_SORTED_INDICES - 1) < numel) {
          if (sorted_indices[idx + num_duplicates + SKIP_SORTED_INDICES - 1] != crnt_sorted_idx) break;
            num_duplicates += SKIP_SORTED_INDICES;
        }
        while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
          num_duplicates++;
        }

        smem_dups_cache[smem_offset + laneIdx] = num_duplicates;
      }
    }

    WARP_SYNC();

    // All lanes in the warp are still active here. Use them all to reduce duplicates when
    // large number of duplicates are present:
    for (int subwarp = 0; subwarp < C10_WARP_SIZE; subwarp++) {
      // All lanes read the shared memory entry for number of duplicates
      int64_t new_num_duplicates = smem_dups_cache[smem_offset + subwarp];

      // Check if the original sub-warp had duplicates to eliminate, if not skip.
      if (new_num_duplicates == 0)
        continue;

      // There are duplicates that need eliminating:
      int64_t new_idx = base_idx + subwarp;
      int64_t new_crnt_sorted_idx = sorted_indices[new_idx];
      const int64_t new_weight_row = new_crnt_sorted_idx * stride + z * stride_before;

      if (!accumulate) {
        const int64_t grad_row = ((int64_t)indices[new_idx + new_num_duplicates - 1]) * stride + z * numel * stride;
        int64_t feature_dim = blockIdx.y * blockDim.x + threadIdx.x;
        while (feature_dim < stride) {
          grad_weight[new_weight_row + feature_dim] = grad_output[grad_row + feature_dim];
          feature_dim += gridDim.y * blockDim.x;
        }
        continue;
      }

      for (int dup = 0; dup < new_num_duplicates; dup++) {
        const int64_t grad_row = ((int64_t) indices[new_idx + dup]) * stride + z * numel * stride;

        // All lanes do the same thing up to here.
        int64_t feature_dim = blockIdx.y * blockDim.x + threadIdx.x;

        // Each lane has a different feature_dim.
        while (feature_dim < stride) {
          grad_weight[new_weight_row + feature_dim] += grad_output[grad_row + feature_dim];
          feature_dim += gridDim.y * blockDim.x;
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void indexing_backward_kernel_stride_1(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  int laneIdx = threadIdx.x % C10_WARP_SIZE;

  const opmath_t scale = (opmath_t)1.0;
  int64_t grad_row = 0;

  extern __shared__ unsigned char smem[];
  auto smem_dups_cache = reinterpret_cast<int64_t*>(smem);

  // Each warp gets a different section of the share memory allocation:
  int smem_offset = threadIdx.y * C10_WARP_SIZE;

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z) {
    // Init duplicates every time we compute a new set of entries:
    smem_dups_cache[smem_offset + laneIdx] = 0;

    int64_t base_idx = blockIdx.x * blockDim.y * C10_WARP_SIZE + threadIdx.y * C10_WARP_SIZE;
    int64_t idx = base_idx + laneIdx;

    // Each lane calculates the number of duplicates:
    if (idx < numel) {
      int64_t crnt_sorted_idx = sorted_indices[idx];

      if (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]) {
        // Determine the number of duplicates in advance:
        int64_t num_duplicates = 1;

        // Lookahead in case there is a large number of duplicates. Once that is done, handle the tail.
        while ((idx + num_duplicates + SKIP_SORTED_INDICES - 1) < numel) {
          if (sorted_indices[idx + num_duplicates + SKIP_SORTED_INDICES - 1] != crnt_sorted_idx) break;
            num_duplicates += SKIP_SORTED_INDICES;
        }
        while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
          num_duplicates++;
        }

        if (!accumulate) {
          const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
          grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride + z * numel * stride;
          grad_weight[weight_row] =
            static_cast<scalar_t>(static_cast<opmath_t>(grad_output[grad_row]) * scale);
          continue;
        }

        // Each lane sequentially handles the duplicate elimination:
        if (num_duplicates < C10_WARP_SIZE) {
          opmath_t gradient = (opmath_t)0.0;
          const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
          for (int64_t i = 0; i < num_duplicates; ++i) {
            grad_row = ((int64_t) indices[idx + i]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
          }

          grad_weight[weight_row] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[weight_row]) + gradient);
        } else {
          // Add duplicate to the cache:
          smem_dups_cache[smem_offset + laneIdx] = num_duplicates;
        }
      }
    }

    WARP_SYNC();

    // All lanes in the warp are still active here. Use them all to reduce duplicates when
    // large number of duplicates are present:
    for (int subwarp = 0; subwarp < C10_WARP_SIZE; subwarp++) {
      // All lanes read the shared memory entry for number of duplicates
      int64_t new_num_duplicates = smem_dups_cache[smem_offset + subwarp];

      // Check if the original sub-warp had duplicates to eliminate, if not skip.
      if (new_num_duplicates == 0)
        continue;

      // There are duplicates that need eliminating:
      int64_t new_idx = base_idx + subwarp;
      int64_t new_crnt_sorted_idx = sorted_indices[new_idx];
      const int64_t new_weight_row = new_crnt_sorted_idx * stride + z * stride_before;

      // Result of the reduction will be in this variable:
      opmath_t gradient = (opmath_t)0.0;

      int64_t num_warp_passes = new_num_duplicates / C10_WARP_SIZE;
      // Parallel reduction across the array of duplicates using all the lanes in the warp:
      for (int64_t i = 0; i < num_warp_passes; ++i) {
        grad_row = ((int64_t) indices[new_idx + i * C10_WARP_SIZE + laneIdx]) * stride + z * numel * stride;
        gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
      }

      // Reduce across the lanes of the warp:
      WARP_SYNC();
      for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
        gradient += WARP_SHFL_DOWN(gradient, offset);
      }

      if (laneIdx == 0) {
        for (int64_t i = num_warp_passes * C10_WARP_SIZE; i < new_num_duplicates; ++i) {
          grad_row = ((int64_t) indices[new_idx + i]) * stride + z * numel * stride;
          gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
        }

        grad_weight[new_weight_row] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[new_weight_row]) + gradient);
      }
    }
  }
}
#endif

template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
//numel is total number of flattened indices, not expanded to dimensions that are not indexed.
//stride is the cumulative size of the not-indexed last dimensions
//stride_before is the stride of the dimension immediately preceding first indexed dimension
//if indexing starts from the 0th dimension, stride_before does not matter because blockIdx.z will be 0 in this case
//outer_dim is number of elements in the first unindexed dimensions
  using opmath_t = at::opmath_type<scalar_t>;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same destination index as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        // if not accumulate, we only keep the last duplicate index so skip those before it
        if (!accumulate && (idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
              if (accumulate) {
                weight[ii] = static_cast<opmath_t>(grad_weight[weight_row + feature_dim]);
              }
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            if (accumulate) {
              weight[ii] += gradient[ii] * scale;
            } else {
              weight[ii] = gradient[ii] * scale;
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(weight[ii]);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}

#ifndef USE_ROCM
template <typename scalar_t>
__global__ void indexing_backward_kernel_stride_1(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t crnt_sorted_idx = sorted_indices[idx];

    if ((idx < numel) &&
        (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]))
    {
      // Determine the number of duplicates in advance
      int64_t num_duplicates = 1;
      while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
        num_duplicates++;
      }

      // Continue computing weights
      const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
      int64_t grad_row = 0;
      const opmath_t scale = (opmath_t)1.0;

      if (!accumulate) {
        grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride + z * numel * stride;
        grad_weight[weight_row] =
          static_cast<scalar_t>(static_cast<opmath_t>(grad_output[grad_row]) * scale);
      } else {
        opmath_t gradient = (opmath_t)0.0;

        int laneIdx = threadIdx.x % C10_WARP_SIZE;
        int64_t num_warp_passes = num_duplicates / C10_WARP_SIZE;
        for (int64_t i = 0; i < num_warp_passes; ++i) {
            grad_row = ((int64_t) indices[idx + i * C10_WARP_SIZE + laneIdx]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
        }
        WARP_SYNC();
        for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
          gradient += WARP_SHFL_DOWN(gradient, offset);
        }

        if (laneIdx == 0) {
          for (int64_t i = num_warp_passes * C10_WARP_SIZE; i < num_duplicates; ++i) {
            grad_row = ((int64_t) indices[idx + i]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
          }

          grad_weight[weight_row] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[weight_row]) + gradient);
        }
      }
    }
  }
}
#endif

template <typename scalar_t>
__global__ void indexing_backward_kernel_small_stride(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t tidx = threadIdx.x;
    int64_t crnt_sorted_idx = sorted_indices[idx];

    if ((idx < numel) &&
        (tidx < stride) &&
        (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]))
    {
      // Determine the number of duplicates in advance
      int64_t num_duplicates = 1;
      while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
        num_duplicates++;
      }

      // Continue computing weights
      const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
      int64_t grad_row = 0;
      const opmath_t scale = (opmath_t)1.0;

      if (!accumulate) {
        grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride + z * numel * stride;
        grad_weight[weight_row + tidx] =
          static_cast<scalar_t>(static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale);
      } else {
        opmath_t gradient = (opmath_t)0.0;
        for (int64_t i = 0; i < num_duplicates; ++i) {
          grad_row = ((int64_t) indices[idx + i]) * stride + z * numel * stride;
          gradient += static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale;
        }

        grad_weight[weight_row + tidx] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[weight_row + tidx]) + gradient);
      }
    }
  }
}

template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel_quantized(
  const int64_t* sorted_indices, const int64_t* indices, const float* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim,
  float inv_scale, int zero_point, int64_t qmin, int64_t qmax) {

  // This implementation is adopted from indexing_backward_kernel above.
  using opmath_t = at::opmath_type<float>;
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        // we only keep the last duplicate index so skip those before it
        if ((idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            weight[ii] = gradient[ii] * scale;
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                // we do quantization here
                int64_t qvalue = static_cast<int64_t>(zero_point + nearbyintf(weight[ii]* inv_scale));
                qvalue = min(max(qvalue, qmin), qmax);
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(qvalue);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}


}


namespace at::native {

namespace {

class ReduceMultiply {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMul(self_data_start + index, *src_data);
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
    opportunistic_fastAtomicAdd(self_data_start, index, numel, *src_data);
#else
    fastAtomicAdd(self_data_start, index, numel, *src_data, true);
#endif
  }
};
static ReduceAdd reduce_add;

class ReduceMinimum {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMin(self_data_start + index, *src_data);
  }
};
static ReduceMinimum reduce_minimum;

class ReduceMaximum {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMax(self_data_start + index, *src_data);
  }
};
static ReduceMaximum reduce_maximum;

}

static Tensor wrapIndexOnce(const Tensor & index, int64_t dim, int64_t dim_size, bool check_range=true) {
//we don't need to check range in backward - if there were out of bounds indices forward should already have errored out
  if (index.numel() != 0 && check_range) {
    at::_assert_async(index.max() < dim_size);
    at::_assert_async(index.min() >= -dim_size);
  }
  return index.remainder(dim_size);
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contiguous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  if (stride.empty()) {
    return stride;
  }
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1, std::multiplies<int64_t>());
  return stride;
}

static std::tuple<Tensor, int64_t, int64_t, int64_t, int64_t, int64_t>
computeLinearIndex(const Tensor & src, TensorList indices, bool check_range) {
  auto strides = computeLinearStride(src);
  const auto& device = src.options().device();

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t nElemBefore = 1, nElemAfter = 1, strideBefore =0;
  int64_t dims_before = 0, dims_indexed = 0;
  for (const auto i: c10::irange(src.dim())) {
    if (indices[i].defined()) {
      dims_indexed++;
      // Cast index to the longType matching src's device
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, src.size(i), check_range) * strides[i]).to(device);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
        if (i>0) {
           strideBefore = src.stride(i-1); // stride after undefined dimensions
        }
      }
    } else if (linearIndex.defined()) {
      nElemAfter *= src.size(i);
    } else {
      dims_before++;
      nElemBefore *= src.size(i);
    }
  }

  return std::make_tuple(std::move(linearIndex), nElemBefore, strideBefore, nElemAfter, dims_before, dims_indexed);
}


static std::tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>, int64_t, int64_t>
makeLinearIndex(Tensor self, IOptTensorListRef orig, bool check_range) {
  checkIndexTensorTypes(orig, /*allow_int*/true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  for (auto & i : indices) {
    if (i.defined() && i.dtype() == at::kInt) {
      i = i.to(at::kLong);
    }
  }
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  std::vector<int64_t> inversePerm;
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices, inversePerm) = transposeToFrontAndInvPerm(self, indices);
  }
  auto [linearIndex, nElemBefore, strideBefore, nElemAfter, dims_before, dims_indexed] =
    computeLinearIndex(self, indices, check_range);
  return std::make_tuple(linearIndex, self, nElemBefore, strideBefore, nElemAfter, inversePerm,
                         dims_before, dims_indexed);
}
namespace {

int64_t largestIndex(const Tensor &self) {
  int64_t result = 0;
  for (const auto i: c10::irange(self.dim())) {
    result += (self.sizes()[i] - 1) * self.strides()[i];
  }
  return result;
}

DimVector valsShape(IntArrayRef self_sizes,
                              int64_t dims_before,
                              int64_t dims_indexed,
                              IntArrayRef replacement_shape) {
  auto shape = DimVector(self_sizes);
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  shape.insert(
    shape.begin() + dims_before,
    replacement_shape.begin(),
    replacement_shape.end());
  return shape;
}

void index_put_with_sort_kernel(Tensor & self, const c10::List<std::optional<Tensor>>& indices, const Tensor & value, bool accumulate, bool unsafe) {
  TORCH_CHECK(!indices.empty() || is_expandable_to(value.sizes(), self.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", self.sizes());
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize, dims_before, dims_indexed;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm,
  dims_before, dims_indexed) = makeLinearIndex(self_, indices, !unsafe);
  auto vals_shape = valsShape(src.sizes(), dims_before, dims_indexed, linearIndex.sizes());
  int64_t num_indices = linearIndex.numel();
  expandedValue = expandedValue.expand(vals_shape).contiguous();

  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");

      // Sort the inputs into sorted with the corresponding indices
      auto range = at::arange(num_indices, linearIndex.options());
      // linearIndex can not be negative, and we take advantage of this
      // fact to sort on less bits for better performance.
      int64_t nbits = cuda::cub::get_num_bits(largestIndex(self_) / sliceSize);
      cuda::cub::radix_sort_pairs(
        linearIndex.const_data_ptr<int64_t>(), sorted_indices.mutable_data_ptr<int64_t>(),
        range.const_data_ptr<int64_t>(), orig_indices.mutable_data_ptr<int64_t>(),
        num_indices, false, 0, nbits);


      TORCH_INTERNAL_ASSERT(
          linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel(),
          "number of flattened indices did not match number of elements in the value tensor: ",
          linearIndex.numel()*sliceSize*nElemBefore, " vs ", expandedValue.numel());

      const int UNROLL = 4;
      const int indices_per_block = 4;
      const int warp_size = at::cuda::warp_size();
      dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
           std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(warp_size, indices_per_block);

#ifdef USE_ROCM
      dim3 new_grid_many_indices(ceil_div(num_indices, (int64_t) (indices_per_block * warp_size)),
      grid.y == 1 ? std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size))) : grid.y,
      grid.z);
      dim3 new_grid(ceil_div(num_indices, (int64_t) (indices_per_block * warp_size)), grid.y, grid.z);
      size_t smem_dups_size = indices_per_block * warp_size * sizeof(int64_t);
#define KERNEL_GRID new_grid
#define KERNEL_SMEM smem_dups_size
#else
#define KERNEL_GRID grid
#define KERNEL_SMEM 0
#endif

      if (sliceSize == 1) {
        // This implementation is faster with high amounts of duplicates but could overflow
        // if FP16 / BF16 is used
        AT_DISPATCH_V2(
          expandedValue.scalar_type(),
          "indexing_backward_kernel_stride_1",
          AT_WRAP([&] {
            indexing_backward_kernel_stride_1<scalar_t><<<KERNEL_GRID, block, KERNEL_SMEM, stream>>>
            (
              sorted_indices.const_data_ptr<int64_t>(),
              orig_indices.const_data_ptr<int64_t>(),
              expandedValue.const_data_ptr<scalar_t>(),
              src_.mutable_data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }),
          AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
          // AT_EXPAND(AT_FLOAT8_TYPES),
          // TODO(#113663): clean up accumulation behavior in float8 dtypes, accumulate=True
          // should not be supported here, then reenable AT_FLOAT8_DTYPES
          kFloat8_e4m3fn,
          kFloat8_e5m2,
          kFloat8_e4m3fnuz,
          kFloat8_e5m2fnuz,
          kComplexHalf,
          kHalf,
          kBool,
          kBFloat16);
      } else {
        if (sliceSize <= warp_size) {
          AT_DISPATCH_V2(
            expandedValue.scalar_type(),
            "indexing_backward_kernel_small_stride",
            AT_WRAP([&] {
              indexing_backward_kernel_small_stride<scalar_t><<<grid, block, 0, stream>>>(
                sorted_indices.const_data_ptr<int64_t>(),
                orig_indices.const_data_ptr<int64_t>(),
                expandedValue.const_data_ptr<scalar_t>(),
                src_.mutable_data_ptr<scalar_t>(),
                num_indices,
                sliceSize,
                strideBefore,
                nElemBefore,
                accumulate);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }),
            AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
            // AT_EXPAND(AT_FLOAT8_TYPES),
            // TODO(#113663): clean up accumulation behavior in float8 dtypes, accumulate=True
            // should not be supported here, then reenable AT_FLOAT8_DTYPES
            kFloat8_e4m3fn,
            kFloat8_e5m2,
            kFloat8_e4m3fnuz,
            kFloat8_e5m2fnuz,
            kComplexHalf,
            kHalf,
            kBool,
            kBFloat16);
        } else {
#ifdef USE_ROCM
          if (num_indices >= 200000)
            AT_DISPATCH_V2(
              expandedValue.scalar_type(),
              "indexing_backward_many_indices",
              AT_WRAP([&] {
                indexing_backward_kernel_many_indices<scalar_t, UNROLL><<<new_grid_many_indices, block, smem_dups_size, stream>>>(
                  sorted_indices.const_data_ptr<int64_t>(),
                  orig_indices.const_data_ptr<int64_t>(),
                  expandedValue.const_data_ptr<scalar_t>(),
                  src_.mutable_data_ptr<scalar_t>(),
                  num_indices,
                  sliceSize,
                  strideBefore,
                  nElemBefore,
                  accumulate);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }),
              AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
              // AT_EXPAND(AT_FLOAT8_TYPES),
              // TODO(#113663): clean up accumulation behavior in float8 dtypes, accumulate=True
              // should not be supported here, then reenable AT_FLOAT8_DTYPES
              kFloat8_e4m3fn,
              kFloat8_e5m2,
              kFloat8_e4m3fnuz,
              kFloat8_e5m2fnuz,
              kComplexHalf,
              kHalf,
              kBool,
              kBFloat16);
          else
#endif
          AT_DISPATCH_V2(
            expandedValue.scalar_type(),
            "indexing_backward",
            AT_WRAP([&] {
              indexing_backward_kernel<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
                sorted_indices.const_data_ptr<int64_t>(),
                orig_indices.const_data_ptr<int64_t>(),
                expandedValue.const_data_ptr<scalar_t>(),
                src_.mutable_data_ptr<scalar_t>(),
                num_indices,
                sliceSize,
                strideBefore,
                nElemBefore,
                accumulate);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }),
            AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
            // AT_EXPAND(AT_FLOAT8_TYPES),
            // TODO(#113663): clean up accumulation behavior in float8 dtypes, accumulate=True
            // should not be supported here, then reenable AT_FLOAT8_DTYPES
            kFloat8_e4m3fn,
            kFloat8_e5m2,
            kFloat8_e4m3fnuz,
            kFloat8_e5m2fnuz,
            kComplexHalf,
            kHalf,
            kBool,
            kBFloat16);
        }
      }

#undef KERNEL_GRID
#undef KERNEL_SMEM

      if (permuted) {
        self.copy_(src_.permute(inversePerm));
      } else if (!self_contiguous) {
        self.copy_(self_);
      }
  }
}

REGISTER_CUDA_DISPATCH(index_put_with_sort_stub, &index_put_with_sort_kernel)

void index_put_with_sort_quantized(Tensor & self, const c10::List<std::optional<Tensor>>& indices, const Tensor & value, double scale, int zero_point, bool unsafe) {
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize, dims_before, dims_indexed;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm,
  dims_before, dims_indexed) = makeLinearIndex(self_, indices, !unsafe);
  auto vals_shape = valsShape(src.sizes(), dims_before, dims_indexed, linearIndex.sizes());
  int64_t num_indices = linearIndex.numel();
  expandedValue = expandedValue.expand(vals_shape).contiguous();

  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");

      // Sort the inputs into sorted with the corresponding indices
      auto range = at::arange(num_indices, linearIndex.options());
      // linearIndex can not be negative, and we take advantage of this
      // fact to sort on less bits for better performance.
      int64_t nbits = cuda::cub::get_num_bits(largestIndex(self_) / sliceSize);
      cuda::cub::radix_sort_pairs(
        linearIndex.const_data_ptr<int64_t>(), sorted_indices.mutable_data_ptr<int64_t>(),
        range.const_data_ptr<int64_t>(), orig_indices.mutable_data_ptr<int64_t>(),
        num_indices, false, 0, nbits);


      TORCH_INTERNAL_ASSERT(
          linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel(),
          "number of flattened indices did not match number of elements in the value tensor: ",
          linearIndex.numel()*sliceSize*nElemBefore, " vs ", expandedValue.numel());
      const int UNROLL = 4;
      const int indices_per_block = 4;
      const int warp_size = at::cuda::warp_size();
      dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
           std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(warp_size, indices_per_block);

      AT_DISPATCH_QINT_TYPES(
        src.scalar_type(), "indexing_backward_quantized", [&] {
        constexpr int64_t qmin = std::numeric_limits<typename scalar_t::underlying>::min();
        constexpr int64_t qmax = std::numeric_limits<typename scalar_t::underlying>::max();
        float inv_scale = 1.0f / static_cast<float>(scale);

        indexing_backward_kernel_quantized<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
          sorted_indices.const_data_ptr<int64_t>(),
          orig_indices.const_data_ptr<int64_t>(),
          expandedValue.const_data_ptr<float>(),
          src_.mutable_data_ptr<scalar_t>(),
          num_indices,
          sliceSize,
          strideBefore,
          nElemBefore,
          inv_scale,
          zero_point,
          qmin,
          qmax);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

      if (permuted) {
        self.copy_(src_.permute(inversePerm));
      } else if (!self_contiguous) {
        self.copy_(self_);
      }
  }
}

REGISTER_CUDA_DISPATCH(index_put_with_sort_quantized_stub, &index_put_with_sort_quantized)
} //anonymous


// Check tensor dimensions for index operations, and return the slice size.
static size_t getSliceSize(const Tensor & dst,
                              int dim,
                              const Tensor & index,
                              const Tensor & src)
{
  const auto dstDims = dst.dim();
  const auto srcDims = src.dim();

  TORCH_CHECK(index.dim() <= 1, "Index must be vector or scalar");

  size_t dstSliceSize = 1;
  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim ", dim, " is out of bounds");
  for (const auto d: c10::irange(dstDims)) {
    if (d != dim) {
      dstSliceSize *= dst.size(d);
    }
  }

  TORCH_CHECK(dim < srcDims, "Indexing dim ", dim, " is out of bounds");
  TORCH_CHECK(index.numel() == src.size(dim),
             "length of src.size[dim] is not equal to length of indices");

  size_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims) mismatch = true;

  for (const auto d: c10::irange(srcDims)) {
    if (d != dim) {
      srcSliceSize *= src.size(d);
      if (!mismatch && dst.size(d) != src.size(d)) mismatch = true;
    }
  }

  TORCH_CHECK(dstSliceSize == srcSliceSize,
             "Source/destination tensor have different slice sizes (%ld vs %ld)",
             dstSliceSize, srcSliceSize);

  if (mismatch) {
    TORCH_WARN_ONCE(
        "Warning: source/destination slices have same size but different "
        "shape for an index operation.  This behavior is deprecated.\n");
  }

  return dstSliceSize;
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFuncLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          typename func_t>
__global__ void indexFuncSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<const T, IndexType> src,
                                    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src.data[srcOffset] * alpha;
      op(dst.data, dstOffset, dstNumel, &val);
    }

  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFuncSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<const T, IndexType> src,
                                    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType totalSize,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    T val = src.data[srcOffset] * alpha;
    op(dst.data, dstOffset, dstNumel, &val);
  }
}

// Compare the stride between adjacent slices (sliceStride) with strides in the
// other dimensions (i.e., strides *inside* each slice).
//
// - Returns true if some dimension inside the slice has lower stride than
//   sliceStride.  The simplest example is a 2-D contiguous tensor with sliceDim
//   == 0 (that is, each slice is a row).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "index-major order".  For example, if thread count equals slice size, then
//   all threads process slice #0 in lockstep, and then slice #1, and so on.
//
// - Otherwise (i.e., sliceStride has the lowest value), this function returns
//   false.  The simplest example is a 2-D contiguous tensor with sliceDim == 1
//   (each slice is a column).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "elementInSlice-major order".  For example, each thread can process element
//   #0 of every slice, and then element #1 of every slice, and so on.
template <typename scalar_t>
bool indexShouldBeMajor(cuda::detail::TensorInfo<scalar_t, unsigned int> &info,
                                    int sliceDim)
{
  // The stride between adjacent slices (e.g., between element #0 of slice #100
  // and element #0 of slice #101).
  unsigned int sliceStride = info.strides[sliceDim];

  for (const auto i: c10::irange(info.dims)) {
    if (i != sliceDim && info.sizes[i] > 1 && info.strides[i] < sliceStride) {
      return true;
    }
  }

  return false;
}

void index_add_cuda_impl(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }

  // Scalars are treated as 1-d tensor
  const Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(result.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");
  TORCH_CHECK(source.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims" );
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");

  if (globalContext().deterministicAlgorithms()){
    torch::List<std::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for (const auto i: c10::irange(dim)) {
      indices.emplace_back();
    }
    indices.emplace_back(index.to(at::kLong));
    result.index_put_(indices, source * alpha, true);
    return;
  }

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  const uint64_t sliceSize = getSliceSize(self_, dim, index, source_);
  const uint64_t sourceTotalSize = source.numel();
  const uint64_t selfAddDimSize = self_.size(dim);
  const uint64_t numIndex = index.numel();
  const uint64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool indContig = index.is_contiguous();

  const int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)     \
  indexFuncSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>   \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      selfInfo, sourceInfo, indexInfo,                                                  \
      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize,                              \
      selfNumel, reduce_add, alpha_value);                                              \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                        \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)            \
  indexFuncLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                      \
                      SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>          \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                       \
      selfInfo, sourceInfo, indexInfo,                                      \
      selfAddDim, sourceAddDim, sourceTotalSize,                            \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                \
      selfAddDimSize, selfNumel, reduce_add, alpha_value);                  \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  uint64_t defaultMaxBlockThreads = getDefaultMaxThreadsPerBlock();
  const dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (uint64_t)128), (uint64_t)(mpc * 8)));
  const dim3 smallIndexBlock(std::min(sliceSize, (uint64_t)128));

  const dim3 largeIndexGrid(std::min(ceil_div(sourceTotalSize, (uint64_t)128), (uint64_t)(mpc * 8)));
  //On ROCm, std::min -> ::min did not work as expected on when outTotalSize>=2147483648
  dim3 largeIndexBlock( (sourceTotalSize < defaultMaxBlockThreads) ? sourceTotalSize : defaultMaxBlockThreads );

  if (cuda::detail::canUse32BitIndexMath(result) &&
      cuda::detail::canUse32BitIndexMath(source) &&
      cuda::detail::canUse32BitIn
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
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `Indexing.cu_docs.md_docs.md`
- **Keyword Index**: `Indexing.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
