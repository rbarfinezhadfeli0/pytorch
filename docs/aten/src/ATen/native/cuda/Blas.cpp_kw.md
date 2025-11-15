# Keyword Index: `aten/src/ATen/native/cuda/Blas.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/Blas.cpp](../../../../../../aten/src/ATen/native/cuda/Blas.cpp)
- **Documentation**: [`Blas.cpp_docs.md`](./Blas.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Activation`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)

### Functions

- **`_addmm_dtype_cuda`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`_baddbmm_dtype_cuda`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`_bmm_dtype_cuda`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`_int_mm_cuda`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`_mm_dtype_cuda`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`activation_to_gemm_and_blas_arg`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`baddbmm_bmm_out_dtype_checks`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`dot_check`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`dot_cuda`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`if`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`isGloballyDisabledAddmmCudaLt`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`isInputCompliesAddmmCudaLt`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`launchGemmAndBiasCublasLt`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`launchGemmCublas`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`launchTunableGemmAndBias`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`switch`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`vdot_cuda`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ExpandUtils.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/Functions.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/OpMathType.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ceil_div.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/core/NamedTensor.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/cuda/CUDABlas.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/cuda/CUDAScaledBlas.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/cuda/tunable/Tunable.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/cuda/tunable/TunableGemm.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/native/GroupedMMUtils.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/native/Resize.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/native/cuda/GroupMM.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/native/cuda/RowwiseScaledMM.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/native/cuda/ScaledGroupMM.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/native/cuda/cuBlasCommonArgs.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/_addmm_activation_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/_efficientzerotensor.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/_scaled_mm_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/_unsafe_view_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/abs.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/addmm_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/addmv_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/baddbmm_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/bmm_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/copy_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/dot_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/empty.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/empty_strided.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/gelu.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/max.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/mm_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/mul.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/ones.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/relu.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/scalar_tensor_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`ATen/ops/vdot_native.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`c10/core/Scalar.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`c10/core/ScalarType.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`c10/util/Exception.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`c10/util/MaybeOwned.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`c10/util/SmallVector.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`c10/util/typeid.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`cstdint`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`fbgemm_gpu/torch_ops.h`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)

### Namespaces

- **`TORCH_IMPL_FUNC`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`Tensor`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)
- **`at`**: [Blas.cpp_docs.md](./Blas.cpp_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
