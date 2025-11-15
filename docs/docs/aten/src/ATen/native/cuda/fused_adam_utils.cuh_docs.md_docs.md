# Documentation: `docs/aten/src/ATen/native/cuda/fused_adam_utils.cuh_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/fused_adam_utils.cuh_docs.md`
- **Size**: 9,669 bytes (9.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/fused_adam_utils.cuh`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/fused_adam_utils.cuh`
- **Size**: 7,001 bytes (6.84 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/Pow.cuh>
#include <utility>

namespace at::native {

enum class ADAM_MODE : uint8_t { ORIGINAL = 0, ADAMW = 1 };

namespace {

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kExpAvgIdx = 2;
constexpr uint8_t kExpAvgSqIdx = 3;
constexpr uint8_t kMaxExpAvgSqIdx = 4;

template <
    typename scalar_type,
    typename opmath_t,
    int depth,
    ADAM_MODE adam_mode,
    bool amsgrad>
C10_DEVICE inline void adam_math(
    scalar_type r_args[depth][kILP],
    const double& lr,
    const double& beta1,
    const double& beta2,
    const double& weight_decay,
    const double& eps,
    const bool& maximize,
    const float* grad_scale_ptr,
    const float* found_inf_ptr,
    const opmath_t& bias_correction1,
    const opmath_t& bias_correction2_sqrt) {
  static_assert(depth == 4 || depth == 5);
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    // Load values.
    opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
    if (grad_scale_ptr) {
      grad /= (static_cast<double>(*grad_scale_ptr));
    }
    const opmath_t grad_to_store = grad;
    if (maximize) {
      grad = -grad;
    }
    opmath_t exp_avg = static_cast<opmath_t>(r_args[kExpAvgIdx][ii]);
    opmath_t exp_avg_sq = static_cast<opmath_t>(r_args[kExpAvgSqIdx][ii]);
    opmath_t max_exp_avg_sq;
    if (amsgrad) {
      max_exp_avg_sq = static_cast<opmath_t>(r_args[kMaxExpAvgSqIdx][ii]);
    }
    // Update param, grad, 1st and 2nd order momentum.
    if (weight_decay != 0) {
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad += param * weight_decay;
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param -= lr * weight_decay * param;
      }
    }
    // todo(crcrpar): use lerp
    // ref: https://developer.nvidia.com/blog/lerp-faster-cuda/
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;
    const opmath_t step_size = lr / bias_correction1;
    opmath_t denom;
    if (amsgrad) {
      max_exp_avg_sq = std::max(max_exp_avg_sq, exp_avg_sq);
      denom = (std::sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + eps;
    } else {
      denom = (std::sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
    }
    param -= step_size * exp_avg / denom;

    // Store results.
    r_args[kParamIdx][ii] = param;
    if (grad_scale_ptr) {
      r_args[kGradIdx][ii] = grad_to_store;
    }
    r_args[kExpAvgIdx][ii] = exp_avg;
    r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
    if (amsgrad) {
      r_args[kMaxExpAvgSqIdx][ii] = max_exp_avg_sq;
    }
  }
}

// [note: Conditional Gradient Store when `optimizer.step` is called by
// GradScaler] When a user is training their model(s) with an FP16 AMP recipe,
// parameter updates are done via `grad_scaler.step(optimizer)` instead of
// `optimizer.step()`. For most optimizers, GradScaler unscales gradients on
// behalf of those optimizers. Also, before `.step`, it makes sure that all the
// gradients involved are finite, which incurs a device sync. On the other hand,
// fused optimizers set their member variable of `_step_supports_amp_scaling` to
// `True` in order to remove the device sync above. This means that fused
// optimizers have to have their CUDA kernels (a) unscale gradients and (b) skip
// parameter updates accordingly. To be functionally on par with `torch.optim`
// optimizers and `_multi_tensor` ones, the kernel below writes out gradients
// only when `grad_scale_ptr != nullptr.
template <typename scalar_type, int depth, ADAM_MODE adam_mode, bool amsgrad>
struct FusedAdamMathFunctor {
  static_assert(
      depth == 4 || depth == 5,
      "depth of 4 for Adam, depth of 5 for Adam with AMSGrad.");
  using opmath_t = at::opmath_type<scalar_type>;
  C10_DEVICE __forceinline__ void operator()(
      int64_t chunk_size,
      FusedOptimizerTensorListMetadata<depth>& tl,
      const float* lr_ptr,
      const double& lr,
      const double& beta1,
      const double& beta2,
      const double& weight_decay,
      const double& eps,
      const bool& maximize,
      const float* grad_scale_ptr,
      const float* found_inf_ptr) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    const double lr_double = lr_ptr ? *lr_ptr : lr;

    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }
    const auto [bias_correction1, bias_correction2_sqrt] =
        [&]() -> std::pair<double, double> {
      auto* step_count =
          reinterpret_cast<const float*>(tl.state_steps_addresses[tensor_loc]);
      const auto bias_correction1 = 1 - at::native::pow_(beta1, *step_count);
      const auto bias_correction2 = 1 - at::native::pow_(beta2, *step_count);
      const auto bias_correction2_sqrt = std::sqrt(bias_correction2);
      return {bias_correction1, bias_correction2_sqrt};
    }();

    scalar_type* args[depth];
    scalar_type r_args[depth][kILP];
    const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

    const bool all_aligned{
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
#pragma unroll
        for (int i = 0; i < depth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        adam_math<scalar_type, opmath_t, depth, adam_mode, amsgrad>(
            r_args,
            lr_double,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr,
            bias_correction1,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < depth; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<depth>(r_args, args, i_start, chunk_size, n);
        adam_math<scalar_type, opmath_t, depth, adam_mode, amsgrad>(
            r_args,
            lr_double,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr,
            bias_correction1,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < depth; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            store_args(args[i], r_args[i], i_start, chunk_size, n);
          }
        }
      }
    }
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

- `ATen/core/Tensor.h`
- `ATen/native/cuda/ForeachFunctors.cuh`
- `ATen/native/cuda/MultiTensorApply.cuh`
- `ATen/native/cuda/Pow.cuh`
- `utility`


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

- **File Documentation**: `fused_adam_utils.cuh_docs.md`
- **Keyword Index**: `fused_adam_utils.cuh_kw.md`
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

- **File Documentation**: `fused_adam_utils.cuh_docs.md_docs.md`
- **Keyword Index**: `fused_adam_utils.cuh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
