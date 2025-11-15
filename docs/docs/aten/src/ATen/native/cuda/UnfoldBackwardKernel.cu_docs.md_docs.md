# Documentation: `docs/aten/src/ATen/native/cuda/UnfoldBackwardKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/UnfoldBackwardKernel.cu_docs.md`
- **Size**: 7,807 bytes (7.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/UnfoldBackwardKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/UnfoldBackwardKernel.cu`
- **Size**: 5,084 bytes (4.96 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UnfoldBackward.h>

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

// Note on naming: it is unconventional.
// grad_in does not mean that it is a gradient wrt to input,
// grad_in/grad_out is just an input/output of unfold_backward kernel.
//
// unfold_backward, the algorithm is described in
// /native/cpu/UnfoldBackwardKernel.cpp

namespace at::native {

namespace {

template <int n_threads, int n_elems_per_thread, typename func_t>
C10_LAUNCH_BOUNDS_2(n_threads, n_elems_per_thread)
__global__ void _unfold_backward_elementwise_kernel(int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void _launch_unfold_backward_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
    total_n_elems >= 0 && total_n_elems <= std::numeric_limits<int32_t>::max()
  );

  dim3 block(n_threads);
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  dim3 grid((total_n_elems + total_work_block - 1) / total_work_block);

  auto stream = at::cuda::getCurrentCUDAStream();
  _unfold_backward_elementwise_kernel<n_threads, n_elems_per_thread, func_t>
    <<<grid, block, 0, stream>>>(total_n_elems, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void _unfold_backward_internal_kernel(
  TensorIterator& iter,
  int64_t size,
  int64_t step,
  int64_t grad_in_dim_stride,
  int64_t grad_in_last_dim_stride,
  int64_t grad_in_dim_size,
  int64_t grad_out_dim_stride
) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unfold_backward_internal_kernel<scalar_t>(
        sub_iter,
        size,
        step,
        grad_in_dim_stride,
        grad_in_last_dim_stride,
        grad_in_dim_size,
        grad_out_dim_stride
      );
    }
    return;
  }

  char* __restrict__ grad_out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ grad_in_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* __restrict__ idx_dim_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  auto offset_calc = make_offset_calculator<3>(iter);

  // The algorithm is: for each index in grad_out find
  // the elements contributing to it and sum them up.
  // Note: the algorithm does not require any synchronization.
  auto loop = [=]C10_DEVICE(int i) {
    auto offsets = offset_calc.get(i);

    auto* __restrict__ grad_out_data = reinterpret_cast<scalar_t*>(grad_out_ptr + offsets[0]);
    auto* __restrict__ grad_in_data = reinterpret_cast<scalar_t*>(grad_in_ptr + offsets[1]);

    auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr + offsets[2]);

    // left_fold potentially intersecting with idx_dim
    // is either (idx_dim - size) / step or the next integer.
    int64_t left_fold_idx = (idx_dim > size) ? (idx_dim - size) / step : 0;
    if (!(left_fold_idx * step <= idx_dim && idx_dim < left_fold_idx * step + size)) {
      ++left_fold_idx;
    }

    auto right_fold_idx = idx_dim / step;
    right_fold_idx = (right_fold_idx >= grad_in_dim_size) ?
      (grad_in_dim_size - 1) : right_fold_idx;

    for (auto fold_idx = left_fold_idx; fold_idx <= right_fold_idx; ++fold_idx) {
      auto idx_last_dim = idx_dim - fold_idx * step;
      *grad_out_data += grad_in_data[fold_idx * grad_in_dim_stride
                                  + idx_last_dim * grad_in_last_dim_stride];
    }

  };

  _launch_unfold_backward_kernel<num_threads(), thread_work_size()>(iter.numel(), loop);
}

void unfold_backward_cuda_kernel(
  Tensor& grad_out,
  const Tensor& grad_in,
  int64_t dim,
  int64_t size,
  int64_t step
) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds
  auto last_dim = maybe_wrap_dim(-1, grad_in.dim());

  auto grad_in_dim_stride = ensure_nonempty_stride(grad_in, dim);
  auto grad_in_last_dim_stride = ensure_nonempty_stride(grad_in, last_dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);

  auto grad_out_dim_stride = ensure_nonempty_stride(grad_out, dim);

  TensorIterator iter = _make_unfold_backward_iter_over_grad_out(
      grad_out, grad_in, dim, size, step);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(),
    "unfold_backward_cuda", [&] {
      _unfold_backward_internal_kernel<scalar_t>(
        iter,
        size,
        step,
        grad_in_dim_stride,
        grad_in_last_dim_stride,
        grad_in_dim_size,
        grad_out_dim_stride
      );
    }
  );
}

}

REGISTER_DISPATCH(unfold_backward_stub, &unfold_backward_cuda_kernel)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/UnfoldBackward.h`
- `ATen/Dispatch.h`
- `ATen/native/cuda/Loops.cuh`
- `ATen/cuda/detail/OffsetCalculator.cuh`
- `ATen/cuda/CUDAContext.h`
- `vector`


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

- **File Documentation**: `UnfoldBackwardKernel.cu_docs.md`
- **Keyword Index**: `UnfoldBackwardKernel.cu_kw.md`
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

- **File Documentation**: `UnfoldBackwardKernel.cu_docs.md_docs.md`
- **Keyword Index**: `UnfoldBackwardKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
