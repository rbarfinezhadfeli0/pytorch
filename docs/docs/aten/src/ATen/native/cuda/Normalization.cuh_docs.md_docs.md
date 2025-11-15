# Documentation: `docs/aten/src/ATen/native/cuda/Normalization.cuh_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/Normalization.cuh_docs.md`
- **Size**: 52,880 bytes (51.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/Normalization.cuh`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Normalization.cuh`
- **Size**: 75,203 bytes (73.44 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <c10/macros/Macros.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// The maximum number of threads in a block
#if defined(USE_ROCM)
constexpr int MAX_BLOCK_SIZE = 1024;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

constexpr unsigned MAX_GRID_SIZE = 65535u;

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(USE_ROCM)
  int threadSizes[5] = { 64, 128, 256, 512, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
  __device__ friend Float2 operator+(Float2 a, const Float2& b) {
    a += b;
    return a;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

template <typename acc_t>
struct SumReduceOp {
    __device__ __forceinline__ acc_t combine(acc_t a, acc_t b) const { return a + b; }

    __device__ __forceinline__ acc_t warp_shfl_down(acc_t data, int offset) const {
        return WARP_SHFL_DOWN(data, offset);
    }
};

template <typename scalar_t, typename accscalar_t>
struct SumReduceOp<Float2<scalar_t, accscalar_t>> {
    using acc_t = Float2<scalar_t, accscalar_t>;

    __device__ __forceinline__ acc_t combine(acc_t a, acc_t b) const { return a + b; }

    __device__ __forceinline__ acc_t warp_shfl_down(acc_t data, int offset) const {
        return {WARP_SHFL_DOWN(data.v1, offset), WARP_SHFL_DOWN(data.v2, offset)};
    }
};

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffling reduction.
// First each warp (of C10_WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most C10_WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than C10_WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
#if defined(USE_ROCM)
    constexpr int UNRL = 4; // load deserilize factor
    scalar_t tmp[UNRL];
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x*UNRL) {
#pragma unroll
      for (int u = 0; u < UNRL; u++)
        tmp[u] = op(batch, plane, std::min((int)tensor.size(2)-1, (int)(x+u*blockDim.x)));
#pragma unroll
      for (int u = 0; u < UNRL; u++)
        if (x+u*blockDim.x < tensor.size(2))
          sum += tmp[u];
    }
#else
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
#endif
  }
  __shared__ scalar_t shared[C10_WARP_SIZE];
  SumReduceOp<scalar_t> reduce_op;
  sum = cuda_utils::BlockReduce<scalar_t, SumReduceOp<scalar_t>, cuda_utils::Block2D>(sum, reduce_op, 0, shared);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
      shared[0] = sum;
  }
  __syncthreads();
  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

constexpr int ELEMENTS_PER_ITER = 4; // enables concurrency within each thread to hide latency
constexpr int ELEMENTS_PER_THREAD = 16;
constexpr int OPTIMAL_TILE_W = 32;
constexpr int MAX_H_BLOCK = 128;

__host__ void flexible_launch_configs(
      const int reduction,
      const int stride,
      dim3 &block,
      dim3 &grid,
      const bool coop_flag = false) {
  int block_x = std::min(lastPow2(stride), OPTIMAL_TILE_W);
  int block_y = std::min(lastPow2(at::ceil_div(reduction , ELEMENTS_PER_THREAD)),
                         MAX_BLOCK_SIZE / block_x);
  if (block_x * block_y != MAX_BLOCK_SIZE) {
    block_x = std::min(lastPow2(stride), MAX_BLOCK_SIZE / block_y);
  }

  int grid_x = at::ceil_div(stride, block_x);
  int grid_y = std::min(at::ceil_div(reduction, block_y * ELEMENTS_PER_THREAD), MAX_H_BLOCK);
  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  block.x = block_x;
  block.y = block_y;
  block.z = 1;
  grid.x = grid_x;
  grid.y = grid_y;
  grid.z = 1;
}

template<typename T, typename C>
__device__ __forceinline__ void welford_merge_element(C& count,
                                                      T& mean,
                                                      T& m2n,
                                                      const C& count_new,
                                                      const T& mean_new,
                                                      const T& m2n_new) {
      T factor = T(1.0) / ::max(1, (count + count_new));
      T delta0 = mean - mean_new;
      mean = (mean_new * count_new + mean * count) * factor;
      m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
      count += count_new;
}

// merge mean/m2n among threadIdx.y within block
template<typename T, typename C>
__device__ __forceinline__ void welford_merge_block_vertical(C& count,
                                                             T& mean,
                                                             T& m2n,
                                                             C* shmem_count,
                                                             T* shmem_mean,
                                                             T* shmem_m2n) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y/2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset*2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];

      welford_merge_element(count, mean, m2n, count_new, mean_new, m2n_new);
    }
  }
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, bool train, typename index_t>
__global__ void batch_norm_transform_input_kernel(
    const GenericPackedTensorAccessor<const input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> output,
    const GenericPackedTensorAccessor<typename std::conditional_t<train, stat_accscalar_t, stat_scalar_t>, 1, RestrictPtrTraits, index_t> mean_,
    const GenericPackedTensorAccessor<typename std::conditional_t<train, stat_accscalar_t, stat_scalar_t>, 1, RestrictPtrTraits, index_t> var_or_invstd,
    const GenericPackedTensorAccessor<const stat_scalar_t, 1, RestrictPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<const stat_scalar_t, 1, RestrictPtrTraits, index_t> bias,
    stat_accscalar_t epsilon) {

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  stat_accscalar_t gamma = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : static_cast<stat_accscalar_t>(1);
  stat_accscalar_t beta = bias.size(0) > 0 ? static_cast<stat_accscalar_t>(bias[plane]) : static_cast<stat_accscalar_t>(0);
  stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
  stat_accscalar_t invstd;
  if (train) {
    invstd = var_or_invstd[plane];
  } else {
    invstd = static_cast<stat_accscalar_t>(1) / device_sqrt(static_cast<stat_accscalar_t>(var_or_invstd[plane]) + epsilon);
  }

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<input_scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
    }
  }
}

struct InvStd {
  template <typename T>
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / device_sqrt(var + epsilon);
    }
    return invstd;
  }
};

struct Var {
  template <typename T>
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    return var;
  }
};

template <typename VarTransform, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_collect_statistics_kernel(
    const GenericPackedTensorAccessor<const input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_mean,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_transformed_var) {

  __shared__ int shared_n[2 * 2 * C10_WARP_SIZE + C10_WARP_SIZE];

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  stat_accscalar_t* shared_avg_var = (stat_accscalar_t*) &shared_n[C10_WARP_SIZE];

  // first the reductions each thread does separately
  stat_accscalar_t avg = 0;
  stat_accscalar_t var_n = 0;
  int n = 0;
  for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
#if defined(USE_ROCM)
    constexpr int UNRL = 4;
    stat_accscalar_t v_[UNRL];
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x*UNRL) {
      for (int u = 0; u < UNRL; u++)
        v_[u] = input[batch][plane][std::min(x+u*blockDim.x, input.size(2)-1)];
      for (int u = 0; u < UNRL; u++) {
        if (x+u*blockDim.x < input.size(2)) {
          stat_accscalar_t d1 = v_[u] - avg;
          n++;
          avg += d1 / n;
          var_n += d1 * (v_[u] - avg);
        }
      }
    }
#else
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
      stat_accscalar_t v = input[batch][plane][x];
      stat_accscalar_t d1 = v - avg;
      n++;
      avg += d1 / n;
      var_n += d1 * (v - avg);
    }
#endif
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, C10_WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, C10_WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, C10_WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % C10_WARP_SIZE == 0) {
    shared_n[tid / C10_WARP_SIZE] = n;
    shared_avg_var[tid / C10_WARP_SIZE * 2] = avg;
    shared_avg_var[tid / C10_WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < C10_WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / C10_WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / C10_WARP_SIZE ? shared_avg_var[2 * tid] : stat_accscalar_t(0));
    var_n = (tid < blockDim.x * blockDim.y  / C10_WARP_SIZE ? shared_avg_var[2 * tid + 1] : stat_accscalar_t(0));
  }
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, C10_WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, C10_WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, C10_WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    if (save_mean.data() != NULL) {
      save_mean[plane] = avg;
    }
    if (save_transformed_var.data() != NULL) {
      save_transformed_var[plane] = VarTransform{}(var_n / N, epsilon);
    }
  }

}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel(
    const GenericPackedTensorAccessor<const input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<const input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_weight,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_bias,
    const GenericPackedTensorAccessor<const stat_scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<const stat_scalar_t, 1, DefaultPtrTraits, index_t> running_mean,
    const GenericPackedTensorAccessor<const stat_scalar_t, 1, DefaultPtrTraits, index_t> running_var,
    const GenericPackedTensorAccessor<const stat_accscalar_t, 1, DefaultPtrTraits, index_t> save_mean,
    const GenericPackedTensorAccessor<const stat_accscalar_t, 1, DefaultPtrTraits, index_t> save_invstd,
    bool train,
    stat_accscalar_t epsilon) {

  index_t plane = blockIdx.x;
  index_t N = grad_output.size(0) * grad_output.size(2);

  stat_accscalar_t mean, invstd;
  if (train) {
    mean = save_mean[plane];
    invstd = save_invstd[plane];
  } else {
    mean = static_cast<stat_accscalar_t>(running_mean[plane]);
    invstd = static_cast<stat_accscalar_t>(1) / device_sqrt(static_cast<stat_accscalar_t>(running_var[plane]) + epsilon);
  }

  stat_accscalar_t weight_val = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : stat_accscalar_t(1);
  stat_accscalar_t norm = stat_accscalar_t(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<input_scalar_t, stat_accscalar_t, GenericPackedTensorAccessor<const input_scalar_t, 3, DefaultPtrTraits, index_t>> g(mean, input, grad_output);
  auto res = reduce<Float2<input_scalar_t, stat_accscalar_t>>(g, grad_output, plane);

  stat_accscalar_t grad_output_sum = res.v1;
  stat_accscalar_t dot_p = res.v2;

  stat_accscalar_t grad_mean = grad_output_sum * norm;
  stat_accscalar_t proj_scale = dot_p * norm * invstd * invstd;
  stat_accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        input_scalar_t go = grad_output[batch][plane][x];
        if (train) {
          stat_accscalar_t inp = input[batch][plane][x];
          stat_accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<input_scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<input_scalar_t>(go * grad_scale);
        }
      }
    }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<stat_scalar_t>(dot_p * invstd);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<stat_scalar_t>(grad_output_sum);
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_reduce_statistics_kernel(
    const GenericPackedTensorAccessor<accscalar_t, 2, RestrictPtrTraits, index_t> vec_mean,
    const GenericPackedTensorAccessor<accscalar_t, 2, RestrictPtrTraits, index_t> vec_invstd,
    GenericPackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t> mean,
    GenericPackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t> invstd,
    GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> running_mean,
    GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> running_var,
    const accscalar_t epsilon,
    const accscalar_t momentum,
    const GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> counts) {

  int feature_size = vec_mean.size(1);
  int world_size = vec_mean.size(0);

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // first the reductions each thread does separately
  for (int i = bid*blockDim.x+tid; i < feature_size; i += gridDim.x*blockDim.x) {
    accscalar_t avg = 0;
    accscalar_t var_n = 0;
    index_t n = 0;
    for (int j = 0; j < world_size; j++) {
      scalar_t count = counts[j];
      accscalar_t m = vec_mean[j][i];
      accscalar_t v = accscalar_t(1.0) / (vec_invstd[j][i]);
      v = (v * v - epsilon) * count;
      accscalar_t factor = 1.0 / (n + count);
      var_n += v + (avg - m) * (avg - m) * n * count * factor;
      avg = n * factor * avg + count * factor * m;
      n += count;
    }
    mean[i] = avg;
    invstd[i] = static_cast<accscalar_t>(1) / device_sqrt(var_n / n + epsilon);
    if (running_mean.data() != NULL) {
      running_mean[i] = static_cast<scalar_t>((1 - momentum) * running_mean[i] + momentum * avg);
    }
    accscalar_t unbiasedVar = var_n / (n - 1);
    if (running_var.data() != NULL) {
      running_var[i] = static_cast<scalar_t>((1 - momentum) * running_var[i] + momentum * unbiasedVar);
    }
  }

}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_reduce_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> invstd,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy_xmu,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_weight,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_bias) {

  index_t plane = blockIdx.x;

  stat_accscalar_t r_mean = mean[plane];
  stat_accscalar_t factor = invstd[plane];

  GradOp<input_scalar_t, stat_accscalar_t, GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>> g(r_mean, input, grad_output);
  auto res = reduce<Float2<input_scalar_t, stat_accscalar_t>>(g, grad_output, plane);

  if (threadIdx.x == 0) {
    if (grad_weight.size(0) > 0) {
      grad_weight[plane] = static_cast<stat_scalar_t>(res.v2 * factor);
    }
    if (grad_bias.size(0) > 0) {
      grad_bias[plane] = static_cast<stat_scalar_t>(res.v1);
    }
    if (sum_dy.size(0) > 0) {
      sum_dy[plane] = static_cast<stat_accscalar_t>(res.v1);
    }
    if (sum_dy_xmu.size(0) > 0) {
      sum_dy_xmu[plane] = static_cast<stat_accscalar_t>(res.v2);
    }
  }
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__device__ __forceinline__ void batch_norm_backward_elemt_kernel_impl(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> invstd,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy_xmu,
    GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    const stat_accscalar_t norm_fct) {
  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  stat_accscalar_t m_c = mean[plane];
  stat_accscalar_t m_dy_c = sum_dy[plane] * norm_fct;
  stat_accscalar_t factor_1_c = invstd[plane];
  stat_accscalar_t factor_2_c = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : stat_accscalar_t(1);
  factor_2_c *= factor_1_c;
  factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu[plane] * norm_fct;

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto g_i = grad_input[batch][plane];
    auto g_o = grad_output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      g_i[feature] = static_cast<input_scalar_t>((g_o[feature] - m_dy_c - (i[feature] - m_c) * factor_1_c) * factor_2_c);
    }
  }
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_elemt_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> invstd,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy_xmu,
    GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    const int* __restrict__ numel, const int world_size) {
  int64_t total_numel = 0;
  for (int i = 0; i < world_size; i ++) {
    total_numel += numel[i];
  }

  const stat_accscalar_t norm_fct =
      static_cast<stat_accscalar_t>(1) / static_cast<stat_accscalar_t>(total_numel);
  batch_norm_backward_elemt_kernel_impl(
      input, grad_output, mean, invstd, weight, sum_dy, sum_dy_xmu, grad_input, norm_fct);
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_elemt_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> invstd,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy_xmu,
    GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    const stat_accscalar_t norm_fct) {
  batch_norm_backward_elemt_kernel_impl(
      input, grad_output, mean, invstd, weight, sum_dy, sum_dy_xmu, grad_input, norm_fct);
}

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> get_packed_accessor(
    const Tensor& t, std::string_view var_name) {
  constexpr auto expect_type = c10::CppTypeToScalarType<typename std::remove_const_t<scalar_t>>::value;
  const auto actual_type = t.scalar_type();
  TORCH_CHECK(actual_type == expect_type, "Expected ", var_name,
              " to have type ", expect_type, " but got ", actual_type);
  return t.generic_packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(
    const Tensor& t, std::string_view var_name) {
  if (!t.defined()) {
    const std::array<index_t, dim> zeros{{0}};
    return GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return get_packed_accessor<scalar_t, dim, PtrTraits, index_t>(t, var_name);
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda_template(const Tensor& grad_out_, const Tensor& input_, const Tensor& weight_,
                                                                     const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
                                                                     bool train, double epsilon, std::array<bool,3> grad_input_mask) {

  using accscalar_t = at::acc_type<stat_scalar_t, true>;
  Tensor grad_input_;
  Tensor grad_input_reshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    grad_input_reshaped = grad_input_.view(input_reshaped.sizes());
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  auto input = get_packed_accessor<
      const input_scalar_t, 3, DefaultPtrTraits, index_t>(input_reshaped, "input");
  auto grad_output = get_packed_accessor<
      const input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_output_reshaped, "grad_output");
  auto grad_input = packed_accessor_or_dummy<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_input_reshaped, "grad_input");
  auto weight = packed_accessor_or_dummy<
      const stat_scalar_t, 1, DefaultPtrTraits, index_t>(weight_, "weight");
  auto grad_weight = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_weight_, "grad_weight");
  auto grad_bias = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_bias_, "grad_bias");
  auto running_mean = packed_accessor_or_dummy<
      const stat_scalar_t, 1, DefaultPtrTraits, index_t>(running_mean_, "running_mean");
  auto running_var = packed_accessor_or_dummy<
      const stat_scalar_t, 1, DefaultPtrTraits, index_t>(running_var_, "running_var");
  auto save_mean = packed_accessor_or_dummy<
      const accscalar_t, 1, DefaultPtrTraits, index_t>(save_mean_, "save_mean");
  auto save_invstd = packed_accessor_or_dummy<
      const accscalar_t, 1, DefaultPtrTraits, index_t>(save_invstd_, "save_invstd");

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

  batch_norm_backward_kernel<input_scalar_t, stat_scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, grad_output, grad_input, grad_weight, grad_bias, weight, running_mean, running_var,
     save_mean, save_invstd, train, epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

template<typename scalar_t, typename index_t, typename VarTransform>
void batch_norm_stats_cuda_template(
    const Tensor& out_mean, const Tensor& out_invstd, const Tensor& input_, double epsilon) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions

  resize_output(out_mean, {n_input});
  resize_output(out_invstd, {n_input});
  auto input = get_packed_accessor<
      const scalar_t, 3, RestrictPtrTraits, index_t>(input_reshaped, "input");
  TORCH_INTERNAL_ASSERT(out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
                        out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(out_mean.dim() == 1 && out_mean.is_contiguous() &&
                        out_mean.sizes()[0]);

  auto mean = packed_accessor_or_dummy<
      accscalar_t, 1, RestrictPtrTraits, index_t>(out_mean, "out_mean");
  auto invstd = packed_accessor_or_dummy<
      accscalar_t, 1, RestrictPtrTraits, index_t>(out_invstd, "out_invstd");
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  batch_norm_collect_statistics_kernel<VarTransform, scalar_t, scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, epsilon, 0.0, mean, invstd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
void batch_norm_elemt_cuda_template(const Tensor& output_, const Tensor& input_, const Tensor& weight_,
                                    const Tensor& bias_, const Tensor& mean_, const Tensor& invstd_) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto output_reshaped = output_.view({input_.size(0), input_.size(1), -1});

  auto input = get_packed_accessor<
      const input_scalar_t, 3, RestrictPtrTraits, index_t>(input_reshaped, "input");
  auto output = get_packed_accessor<
      input_scalar_t, 3, RestrictPtrTraits, index_t>(output_reshaped, "output");
  auto weight = packed_accessor_or_dummy<
    const stat_scalar_t, 1, RestrictPtrTraits, index_t>(weight_, "weight");
  auto bias = packed_accessor_or_dummy<
      const stat_scalar_t, 1, RestrictPtrTraits, index_t>(bias_, "bias");
  auto mean = packed_accessor_or_dummy<
      stat_accscalar_t, 1, RestrictPtrTraits, index_t>(mean_, "mean");
  auto invstd = packed_accessor_or_dummy<
      stat_accscalar_t, 1, RestrictPtrTraits, index_t>(invstd_, "invstd");
  auto stream = at::cuda::getCurrentCUDAStream();

  // NOTE: We use transform_input_kernel in training mode, which ignores epsilon
  const double dummy_epsilon = 1e-5;

  // The input_transform kernel is pointwise, but we need to balance reading parameters (save_var/mean,
  // weight/bias) - which we only do once and have a for loop afterwards - with having many threads and blocks
  // and good occupancy. Quiet likely, we could go with even more blocks than 1024.
  // The various planes are independent, so we use blocks for them.
  int tf = std::max<int>(getNumThreads(input.size(2)/4),
                         std::min<int>(getNumThreads(input.size(2)), 64));
  int tb = std::max<int>(64/tf, 1);
  dim3 blocks_trans(input.size(1), std::max<int>(1, std::min<int>((256*1024)/input.size(1),
                                                                  (input.size(0)+tb-1)/tb)));
  blocks_trans.y = std::min(blocks_trans.y, MAX_GRID_SIZE);
  dim3 threads_trans(tf, tb);
  batch_norm_transform_input_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, true, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
    (input, output, mean, invstd, weight, bias, dummy_epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename scalar_t, typename accscalar_t, typename index_t>
std::tuple<Tensor, Tensor> batch_norm_gather_stats_cuda_template(const Tensor& mean_, const Tensor& invstd_,
                                                                 const Tensor& running_mean_, const Tensor& running_var_,
                                                                 double momentum, double epsilon, const Tensor& counts_) {

  Tensor save_mean_;
  Tensor save_invstd_;

  auto features = mean_.size(1);
  auto input_options = mean_.options();
  if (mean_.scalar_type() == at::ScalarType::Half || mean_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  save_mean_ = at::empty({features}, input_options);
  save_invstd_ = at::empty({features}, input_options);

  auto mean = packed_accessor_or_dummy<
      accscalar_t, 2, RestrictPtrTraits, index_t>(mean_, "mean");
  auto invstd = packed_accessor_or_dummy<
      accscalar_t, 2, RestrictPtrTraits, index_t>(invstd_, "invstd");
  auto running_mean = packed_accessor_or_dummy<
      scalar_t, 1, RestrictPtrTraits, index_t>(running_mean_, "running_mean");
  auto running_var = packed_accessor_or_dummy<
      scalar_t, 1, RestrictPtrTraits, index_t>(running_var_, "running_mean");
  auto counts = packed_accessor_or_dummy<
      scalar_t, 1, RestrictPtrTraits, index_t>(counts_, "counts");

  auto save_mean = get_packed_accessor<
      accscalar_t, 1, RestrictPtrTraits, index_t>(save_mean_, "save_mean");
  auto save_invstd = get_packed_accessor<
      accscalar_t, 1, RestrictPtrTraits, index_t>(save_invstd_, "save_invstd");
  auto stream = at::cuda::getCurrentCUDAStream();

  int block = getNumThreads(features);
  int grid = std::max<int>(1, features/block);
  batch_norm_reduce_statistics_kernel<scalar_t, accscalar_t, index_t> <<<grid, block, 0, stream>>>
      (mean, invstd, save_mean, save_invstd, running_mean, running_var, epsilon, momentum, counts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(save_mean_, save_invstd_);
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_cuda_template(const Tensor& grad_out_, const Tensor& input_,
                                                                                    const Tensor& mean_, const Tensor& invstd_, const Tensor& weight_,
                                                                                    const bool input_g, const bool weight_g, const bool bias_g) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (input_g) {
    sum_dy_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    sum_dy_xmu_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (weight_g) {
    grad_weight_ = at::empty({n_input}, weight_.options());
  }
  if (bias_g) {
    grad_bias_ = at::empty({n_input}, weight_.options());
  }

  auto input = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(input_reshaped, "input");
  auto grad_output = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_output_reshaped, "grad_output");
  auto grad_weight = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_weight_, "grad_weight");
  auto grad_bias = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_bias_, "grad_bias");
  auto mean = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_, "mean");
  auto invstd = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(invstd_, "invstd");
  auto sum_dy = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_, "sum_dy");
  auto sum_dy_xmu = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_xmu_, "sum_dy_xmu");

  auto batch_size = input_reshaped.size(0);
  auto feature_size = input_reshaped.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();

  int warp_size = at::cuda::warp_size();
  int block_y = std::min<int>(lastPow2(batch_size), MAX_BLOCK_SIZE/warp_size);
  // We want block_x to be at least a warp width
  int block_x = std::min<int>(std::max<int>(getNumThreads(feature_size), warp_size), MAX_BLOCK_SIZE/block_y);
  const dim3 block(block_x, block_y);
  const dim3 grid(n_input);

  batch_norm_backward_reduce_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t> <<<grid, block, 0, stream>>>
    (input, grad_output, mean, invstd, sum_dy, sum_dy_xmu, grad_weight, grad_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(sum_dy_, sum_dy_xmu_, grad_weight_, grad_bias_);
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
Tensor batch_norm_backward_elemt_cuda_template(const Tensor& grad_out_, const Tensor& input_,
                                               const Tensor& mean_, const Tensor& invstd_,
                                               const Tensor& weight_, const Tensor& sum_dy_, const Tensor& sum_dy_xmu_) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());
  auto grad_input_reshaped = at::empty_like(input_reshaped, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto input = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(input_reshaped, "input");
  auto grad_input = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_input_reshaped, "grad_input");
  auto grad_output = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_output_reshaped, "grad_output");
  auto mean = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_, "mean");
  auto invstd = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(invstd_, "invstd");
  auto weight = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(weight_, "weight");
  auto sum_dy = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_, "sum_dy");
  auto sum_dy_xmu = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_xmu_, "sum_dy_xmu");

  auto stream = at::cuda::getCurrentCUDAStream();

  // The kernel is pointwise, but we need to balance reading parameters (save_var/mean,
  // weight/bias) - which we only do once and have a for loop afterwards - with having many threads and blocks
  // and good occupancy. Quiet likely, we could go with even more blocks than 1024.
  // The various planes are independent, so we use blocks for them.
  int tf = std::max<int>(getNumThreads(input.size(2)/4),
                         std::min<int>(getNumThreads(input.size(2)), 64));
  int tb = std::max<int>(64/tf, 1);
  dim3 blocks_trans(input.size(1), std::max<int>(1, std::min<int>((256*1024)/input.size(1),
                                                                  (input.size(0)+tb-1)/tb)));
  blocks_trans.y = std::min(blocks_trans.y, MAX_GRID_SIZE);
  dim3 threads_trans(tf, tb);
  auto reduction_size = input_.numel() / n_input;
  auto norm_fct = static_cast<stat_accscalar_t>(1.0 / reduction_size);
  batch_norm_backward_elemt_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t>
      <<<blocks_trans, threads_trans, 0, stream>>>
      (input, grad_output, mean, invstd, weight, sum_dy, sum_dy_xmu, grad_input, norm_fct);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grad_input_reshaped.view(input_.sizes());
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
Tensor batch_norm_backward_elemt_cuda_template(const Tensor& grad_out_, const Tensor& input_,
                                               const Tensor& mean_, const Tensor& invstd_,
                                               const Tensor& weight_, const Tensor& sum_dy_, const Tensor& sum_dy_xmu_, const Tensor& count) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());
  auto grad_input_reshaped = at::empty_like(input_reshaped, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto input = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(input_reshaped, "input");
  auto grad_input = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_input_reshaped, "grad_input");
  auto grad_output = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_output_reshaped, "grad_output");
  auto mean = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_, "mean");
  auto invstd = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(invstd_, "invstd");
  auto weight = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(weight_, "weight");
  auto sum_dy = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_, "sum_dy");
  auto sum_dy_xmu = packed_accessor_or_dummy<
      stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_xmu_, "sum_dy_xmu");

  auto stream = at::cuda::getCurrentCUDAStream();

  // The kernel is pointwise, but we need to balance reading parameters (save_var/mean,
  // weight/bias) - which we only do once and have a for loop afterwards - with having many threads and blocks
  // and good occupancy. Quiet likely, we could go with even more blocks than 1024.
  // The various planes are independent, so we use blocks for them.
  int tf = std::max<int>(getNumThreads(input.size(2)/4),
                         std::min<int>(getNumThreads(input.size(2)), 64));
  int tb = std::max<int>(64/tf, 1);
  dim3 blocks_trans(input.size(1), std::max<int>(1, std::min<int>((256*1024)/input.size(1),
                                                                  (input.size(0)+tb-1)/tb)));
  blocks_trans.y = std::min(blocks_trans.y, MAX_GRID_SIZE);
  dim3 threads_trans(tf, tb);
  batch_norm_backward_elemt_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
    (input, grad_output, mean, invstd, weight, sum_dy, sum_dy_xmu, grad_input, count.const_data_ptr<int>(), count.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grad_input_reshaped.view(input_.sizes());
}

// welford kernel for c last tensor calculating mean/biased_variance/unbiased_variance
// original apex name: welford_kernel_c_last
template
   <typename VarTransform,
    typename scalar_t,
    typename accscalar_t,
    int PARALLEL_LOADS>
__global__ void
batch_norm_collect_statistics_channels_last_kernel(
      const scalar_t* __restrict__ input,
      accscalar_t* __restrict__ out_mean,
      accscalar_t* __restrict__ out_invstd,
      volatile accscalar_t* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride,
      accscalar_t epsilon) {
  // hide latency with concurrency
  accscalar_t x_mean[PARALLEL_LOADS];
  accscalar_t m_2_n[PARALLEL_LOADS];
  int count[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    x_mean[i] = accscalar_t(0);
    m_2_n[i] = accscalar_t(0);
    count[i] = accscalar_t(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
    accscalar_t x_math[PARALLEL_LOADS];
    accscalar_t x_count_inv[PARALLEL_LOADS];
    accscalar_t is_valid[PARALLEL_LOADS];

    // load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_math[j] = input[address_base];
        count[j]++;
        x_count_inv[j] = accscalar_t(1) / count[j];
        is_valid[j] = accscalar_t(1);
      } else {
        x_math[j] = accscalar_t(0);
        x_count_inv[j] = accscalar_t(0);
        is_valid[j] = accscalar_t(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

    // calculate mean/m2n with welford
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      accscalar_t delta0 = x_math[j] - x_mean[j];
      x_mean[j] += delta0 * x_count_inv[j];
      accscalar_t delta1 = x_math[j] - x_mean[j];
      m_2_n[j] += delta0 * delta1 * is_valid[j];
    }
  }

  // thread reduction to accumulate mean/m_2_n/count between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    welford_merge_element(count[0], x_mean[0], m_2_n[0], count[j], x_mean[j], m_2_n[j]);
  }

  // release x_mean / m_2_n
  auto mean_th = x_mean[0];
  auto m2_th = m_2_n[0];
  auto count_th = count[0];

  // block-wise reduction with shared memory (since reduction cannot be done within a warp)
  static __shared__ accscalar_t shmem_mean[MAX_BLOCK_SIZE];
  static __shared__ accscalar_t shmem_m2n[MAX_BLOCK_SIZE];
  static __shared__ int shmem_count[MAX_BLOCK_SIZE];

  welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count, shmem_mean, shmem_m2n);

  if (gridDim.y > 1) {
    volatile accscalar_t* staging_mean = staging_data;
    volatile accscalar_t* staging_m2n = &staging_data[stride*gridDim.y];
    volatile int* staging_count = reinterpret_cast<volatile int*>(&staging_m2n[stride*gridDim.y]);

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_mean[address_base] = mean_th;
      staging_m2n[address_base] = m2_th;
      staging_count[address_base] = count_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y-1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      count_th = 0;
      mean_th = accscalar_t(0.0);
      m2_th = accscalar_t(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        int count_new = c_offset < stride ? staging_count[address_base] : 0;
        accscalar_t mean_new = c_offset < stride ? staging_mean[address_base] : accscalar_t(0.0);
        accscalar_t m2n_new = c_offset < stride ? staging_m2n[address_base] : accscalar_t(0.0);

        welford_merge_element(count_th, mean_th, m2_th, count_new, mean_new, m2n_new);
      }

      welford_merge_block_vertical(count_th, mean_th,
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

- **File Documentation**: `Normalization.cuh_docs.md_docs.md`
- **Keyword Index**: `Normalization.cuh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
