# Documentation: `docs/aten/src/ATen/native/cuda/Repeat.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/Repeat.cu_docs.md`
- **Size**: 4,889 bytes (4.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/Repeat.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Repeat.cu`
- **Size**: 2,224 bytes (2.17 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Repeat.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/repeat_interleave_native.h>
#endif

template <typename index_t>
__global__ static void compute_cuda_kernel(
    const index_t* repeat_ptr,
    const int64_t* cumsum_ptr,
    index_t* result_ptr,
    int64_t size,
    int64_t result_size) {
  CUDA_KERNEL_ASSERT_PRINTF(
      result_size == cumsum_ptr[size - 1],
      "Invalid input! In `repeat_interleave`, the `output_size` argument (%ld) must be the same as the sum of the elements in the `repeats` tensor (%ld).\n",
      result_size,
      cumsum_ptr[size - 1]);

  int64_t idx = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = (blockDim.x * gridDim.x) / C10_WARP_SIZE;
  int warp_id = idx / C10_WARP_SIZE;
  int tid_in_warp = idx % C10_WARP_SIZE;
  for (int64_t i = warp_id; i < size; i += stride) {
    int64_t end = cumsum_ptr[i];
    index_t repeat = repeat_ptr[i];
    CUDA_KERNEL_ASSERT(repeat >= 0);
    int64_t start = end - repeat;
    for (int64_t j = start + tid_in_warp; j < end; j += C10_WARP_SIZE) {
      result_ptr[j] = i;
    }
  }
}

template <typename index_t>
static void compute_cuda(
    const index_t* repeat_ptr,
    const int64_t* cumsum_ptr,
    index_t* result_ptr,
    int64_t size,
    int64_t result_size) {
  int64_t block = 512;
  int64_t warps_per_block = block / at::cuda::warp_size();
  int64_t grid =
      std::min<int64_t>((size + warps_per_block - 1) / warps_per_block, 2048L);

  compute_cuda_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      repeat_ptr, cumsum_ptr, result_ptr, size, result_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace at::native {

Tensor repeat_interleave_cuda(
    const Tensor& repeat,
    std::optional<int64_t> output_size) {
  Tensor output;
  AT_DISPATCH_INDEX_TYPES(
      repeat.scalar_type(), "repeat_interleave_cuda", [&]() {
        output = repeat_interleave_common<index_t, compute_cuda<index_t>>(
            repeat, output_size);
      });
  return output;
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

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/cuda/CUDAContext.h`
- `ATen/native/Repeat.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/repeat_interleave_native.h`


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

- **File Documentation**: `Repeat.cu_docs.md`
- **Keyword Index**: `Repeat.cu_kw.md`
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

- **File Documentation**: `Repeat.cu_docs.md_docs.md`
- **Keyword Index**: `Repeat.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
