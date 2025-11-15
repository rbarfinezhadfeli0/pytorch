# Documentation: `docs/aten/src/ATen/native/cuda/GroupMMCommon.cuh_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/GroupMMCommon.cuh_docs.md`
- **Size**: 8,996 bytes (8.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/GroupMMCommon.cuh`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/GroupMMCommon.cuh`
- **Size**: 6,452 bytes (6.30 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once
#include <cutlass/util/packed_stride.hpp>

namespace at::cuda::detail {

using Strides = std::array<int64_t, 3>;

template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOutput,
    typename DtypeScale,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideOutput>
__global__ void prepare_grouped_gemm_data(
    DtypeA* A,
    DtypeB* B,
    DtypeOutput* output,
    DtypeScale* scale_A,
    DtypeScale* scale_B,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    DtypeScale** inputA_scale_ptrs,
    DtypeScale** inputB_scale_ptrs,
    ProblemShape* problem_sizes,
    // Strides for cutlass, cute::Stride
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    const int32_t* offs,
    int32_t M,
    int32_t N,
    int32_t K,
    // Original strides of the input tensors
    Strides tensor_StrideA,
    Strides tensor_StrideB,
    Strides tensor_StrideOutput,
    Strides tensor_ShapeA,
    Strides tensor_ShapeB,
    int64_t a_scale_stride,
    int64_t b_scale_stride,
    bool a_row_major = true,
    bool b_row_major = false) {
  int32_t tid = threadIdx.x;
  int32_t delta = 0;
  int32_t offset = 0;
  if (offs != nullptr) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    offset = offs[tid];
    delta = offset - start;
    CUDA_KERNEL_ASSERT(delta >=0 && "expected gemm dimension to be greater or equal 0\n");

    // TMA transfers require global memory tensor addresses to be
    // aligned to 16 bytes.
    if (tid < blockDim.x - 1) {
      // Check this requirement for input tensors, in case group
      // addresses are increased along the dynamic dimension.
      if ((K < 0 && a_row_major) ||       // 2D/2D: check along K dimension
          (M < 0 && !a_row_major)) {      // 3D/2D: check along N dimension
        int align = 128 / cutlass::sizeof_bits<DtypeA>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
      if ((K < 0 && !b_row_major) ||      // 2D/2D: check along K dimension
          (N < 0 && b_row_major)) {       // 3D/2D: check along N dimension
        int align = 128 / cutlass::sizeof_bits<DtypeB>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }

      // Check the same requirement for output tensor (that is always
      // contiguous, and in row-major layout).
      if (N < 0) {
        int align = 128 / cutlass::sizeof_bits<DtypeOutput>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected output tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
    }
  }
  int64_t lda, ldb, ldoutput;
  if (M < 0) {
    // A and output is 2d
    CUDA_KERNEL_ASSERT(offset <= tensor_ShapeA[0] && "expected offset to be less than tensor size\n");
    M = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = tid == 0 ? scale_A : scale_A + offs[tid - 1];
      inputB_scale_ptrs[tid] = scale_B + tid * b_scale_stride;
    }
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1] * ldoutput;
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
  } else if (N < 0) {
    CUDA_KERNEL_ASSERT(offset <= tensor_ShapeB[1] && "expected offset to be less than tensor size\n");
    N = delta;
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1]; // B is transposed
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[1];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * a_scale_stride;
      inputB_scale_ptrs[tid] = tid == 0 ? scale_B : scale_B + offs[tid - 1];
    }
  } else if (K < 0) {
    CUDA_KERNEL_ASSERT(offset <= tensor_ShapeA[1] && offset <= tensor_ShapeB[0] && "expected offset to be less than tensor size\n");
    // A, B is 2d, output is 3d
    K = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * M;
      inputB_scale_ptrs[tid] = scale_B + tid * N;
    }
  } else {
    // A, B, output are 3D
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * a_scale_stride;
      inputB_scale_ptrs[tid] = scale_B + tid * b_scale_stride;
    }
  }
  problem_sizes[tid] = ProblemShape(M, N, K);

  // make_cute_packed_stride only replaces one of the stride elements with
  // one the provided values in the shape arguments
  // the indices of the src/dst depend on whether A/B are row-major
  // so constructing shape argument with two similar lda values
  // while it looks non-sensical (and it is a nonsensical shape)
  // is fine for these stride construction purposes - the one that will be used
  // for replacement is correct, the other one is ignored, and we don't have to
  // branch on whether A/B are row-major
  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {lda, lda, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {ldb, ldb, 1});
  stride_output[tid] =
      cutlass::make_cute_packed_stride(StrideOutput{}, {M, ldoutput, 1});
}
} // namespace at::cuda::detail

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

- `cutlass/util/packed_stride.hpp`


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

- **File Documentation**: `GroupMMCommon.cuh_docs.md`
- **Keyword Index**: `GroupMMCommon.cuh_kw.md`
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

- **File Documentation**: `GroupMMCommon.cuh_docs.md_docs.md`
- **Keyword Index**: `GroupMMCommon.cuh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
