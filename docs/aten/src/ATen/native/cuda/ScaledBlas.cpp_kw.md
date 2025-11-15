# Keyword Index: `aten/src/ATen/native/cuda/ScaledBlas.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/ScaledBlas.cpp](../../../../../../aten/src/ATen/native/cuda/ScaledBlas.cpp)
- **Documentation**: [`ScaledBlas.cpp_docs.md`](./ScaledBlas.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`cublasCommonArgs`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)

### Functions

- **`_check_deepseek_support`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`_scaled_mm_allowed_device`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`_scaled_mm_cuda`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`_scaled_mm_cuda_v2`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`_scaled_mm_is_fnuz`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`check_size_stride`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`if`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`is_blockwise_128x128_scaling`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`is_blockwise_1x128_scaling`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`is_blockwise_1x16_scaling`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`is_blockwise_1x32_scaling`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`is_desired_scaling`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`is_rowwise_scaling`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`is_tensorwise_scaling`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ExpandUtils.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/Functions.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/OpMathType.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ceil_div.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/core/NamedTensor.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/cuda/CUDABlas.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/cuda/CUDAScaledBlas.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/cuda/tunable/Tunable.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/cuda/tunable/TunableGemm.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/native/GroupedMMUtils.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/native/Resize.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/native/cuda/GroupMM.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/native/cuda/RowwiseScaledMM.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/native/cuda/ScaledGroupMM.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/native/cuda/cuBlasCommonArgs.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/_addmm_activation_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/_efficientzerotensor.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/_scaled_mm_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/_unsafe_view_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/abs.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/addmm_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/addmv_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/baddbmm_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/bmm_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/copy_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/dot_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/empty.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/empty_strided.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/gelu.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/max.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/mm_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/mul.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/ones.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/relu.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/scalar_tensor_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`ATen/ops/vdot_native.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`c10/core/Scalar.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`c10/core/ScalarType.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`c10/util/Exception.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`c10/util/MaybeOwned.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`c10/util/SmallVector.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`c10/util/typeid.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`cstdint`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`fbgemm_gpu/torch_ops.h`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)

### Namespaces

- **`at`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`fbgemm_gpu`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`scaled_blas`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)
- **`std`**: [ScaledBlas.cpp_docs.md](./ScaledBlas.cpp_docs.md)


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
