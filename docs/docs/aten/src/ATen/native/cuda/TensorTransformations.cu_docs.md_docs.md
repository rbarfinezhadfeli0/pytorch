# Documentation: `docs/aten/src/ATen/native/cuda/TensorTransformations.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/TensorTransformations.cu_docs.md`
- **Size**: 7,895 bytes (7.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/TensorTransformations.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/TensorTransformations.cu`
- **Size**: 5,028 bytes (4.91 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorTransformations.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/roll_native.h>
#endif

#include <cstddef>
#include <vector>

namespace at::native {

template <typename scalar_t, typename IndexType>
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(cuda::getApplyBlockSize(), cuda::getApplyBlocksPerSM())
#endif
__global__ void kernel_pointwise_flip_apply2(
    const cuda::detail::TensorInfo<scalar_t, IndexType> in_tensor_info,
    cuda::detail::TensorInfo<scalar_t, IndexType> out_tensor_info,
    IndexType N,
    int flip_dim,
    IndexType total_dims) {
  for (IndexType linear_index = blockIdx.x * blockDim.x + threadIdx.x; linear_index < N; linear_index += gridDim.x * blockDim.x) {
    IndexType dst_offset = 0;
    if (flip_dim == 0) {
      // flip 1st dim
      dst_offset = (in_tensor_info.sizes[0] - 1 - linear_index / in_tensor_info.strides[0]) * in_tensor_info.strides[0] + linear_index % in_tensor_info.strides[0];
    }
    else {
      // flip last dim
      IndexType i = total_dims - 1;
      dst_offset = linear_index / in_tensor_info.strides[0] * in_tensor_info.strides[0] + (in_tensor_info.sizes[i] - 1 - linear_index % in_tensor_info.strides[0]);
    }
    out_tensor_info.data[dst_offset] = in_tensor_info.data[linear_index];
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(cuda::getApplyBlockSize())
__global__ void flip_cuda_kernel(
    scalar_t* in_tensor,
    scalar_t* out_tensor,
    int64_t N,
    int64_t* flip_dims,
    int64_t flip_dims_size,
    int64_t* strides,
    int64_t* strides_contiguous,
    int64_t* shape,
    int64_t total_dims) {
  int64_t linear_index = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear_index >= N) {
    return;
  }

  int64_t cur_indices = linear_index, rem = 0, dst_offset = 0;
  for (int64_t i = 0; i < total_dims; i++) {
    int64_t temp = cur_indices;
    cur_indices = cur_indices / strides_contiguous[i];
    rem = temp - cur_indices * strides_contiguous[i];
    // flip the indices if it is in flip_dims
    for (int64_t j = 0; j < flip_dims_size; j++) {
      if (i == flip_dims[j]) {
        cur_indices = shape[i] - 1 - cur_indices;
      }
    }
    dst_offset += cur_indices * strides[i];
    cur_indices = rem;
  }
  out_tensor[linear_index] = in_tensor[dst_offset];
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(cuda::getApplyBlockSize())
__global__ void roll_cuda_kernel(
    const scalar_t* in_tensor,
    scalar_t* out_tensor,
    int64_t N,
    int64_t roll_dim,
    int64_t start,
    int64_t size,
    int64_t stride,
    int64_t total_dims) {
  int64_t linear_index = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear_index >= N) {
    return;
  }
  // roll dim idx is the index of linear_index along the rolling dimension.
  int64_t roll_dim_idx = linear_index % (stride * size) / stride;
  // index into the source data to find appropriate value.
  int64_t source_idx = 0;
  if( roll_dim_idx >= (size - start) ) {
    source_idx = linear_index - ((size - start) * stride);
  } else {
    source_idx = linear_index + (start * stride);
  }
  out_tensor[linear_index] = in_tensor[source_idx];
}

// Roll a tensor along a dimension
Tensor roll_cuda(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }

  auto in_tensor = self;
  if(!self.is_contiguous()) {
    in_tensor = self.contiguous();
  }
  auto out_tensor = at::empty_like(in_tensor, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (out_tensor.numel() == 0) {
    return out_tensor;
  }
  const int64_t N = in_tensor.numel();
  const int64_t dim = dims[0];
  const int64_t size = in_tensor.size(dim);
  int64_t start = (size - shifts[0]) % size;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if( start < 0 ) start = start + size;

  dim3 dim_block = cuda::getApplyBlock();
  dim3 dim_grid;
  TORCH_CHECK(cuda::getApplyGrid(N, dim_grid, in_tensor.get_device()), "unable to get dim grid");

  auto total_dims = in_tensor.dim();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      in_tensor.scalar_type(), "roll_cuda",
      [&] {
        roll_cuda_kernel<<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          in_tensor.const_data_ptr<scalar_t>(), out_tensor.mutable_data_ptr<scalar_t>(), N,
          dim, start,
          size,
          in_tensor.stride(dim),
          total_dims);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return out_tensor;
}

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

- `ATen/native/TensorTransformations.h`
- `ATen/Dispatch.h`
- `ATen/cuda/detail/IndexUtils.cuh`
- `ATen/cuda/CUDAApplyUtils.cuh`
- `ATen/cuda/CUDAContext.h`
- `c10/macros/Macros.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/roll_native.h`
- `cstddef`
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

- **File Documentation**: `TensorTransformations.cu_docs.md`
- **Keyword Index**: `TensorTransformations.cu_kw.md`
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

- **File Documentation**: `TensorTransformations.cu_docs.md_docs.md`
- **Keyword Index**: `TensorTransformations.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
