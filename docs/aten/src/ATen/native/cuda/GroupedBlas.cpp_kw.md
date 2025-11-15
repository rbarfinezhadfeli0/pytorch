# Keyword Index: `aten/src/ATen/native/cuda/GroupedBlas.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/GroupedBlas.cpp](../../../../../../aten/src/ATen/native/cuda/GroupedBlas.cpp)
- **Documentation**: [`GroupedBlas.cpp_docs.md`](./GroupedBlas.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_check_scales_blocked`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`_check_scales_fp8_rowwise`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`_grouped_mm_cuda`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`_scaled_grouped_mm_cuda`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`_scaled_grouped_mm_cuda_v2`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`check_scale`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`if`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ExpandUtils.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/Functions.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/OpMathType.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ceil_div.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/core/NamedTensor.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/cuda/CUDABlas.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/cuda/CUDAScaledBlas.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/cuda/tunable/Tunable.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/cuda/tunable/TunableGemm.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/native/GroupedMMUtils.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/native/Resize.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/native/cuda/GroupMM.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/native/cuda/RowwiseScaledMM.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/native/cuda/ScaledGroupMM.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/native/hip/ck_group_gemm.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/_addmm_activation_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/_efficientzerotensor.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/_scaled_mm_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/_unsafe_view_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/abs.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/addmm_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/addmv_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/baddbmm_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/bmm_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/copy_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/dot_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/empty.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/empty_strided.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/gelu.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/max.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/mm_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/mul.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/ones.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/relu.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/scalar_tensor_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`ATen/ops/vdot_native.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`c10/core/Scalar.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`c10/core/ScalarType.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`c10/util/Exception.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`c10/util/MaybeOwned.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`c10/util/SmallVector.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`c10/util/typeid.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`cstdint`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`fbgemm_gpu/torch_ops.h`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)

### Namespaces

- **`Tensor`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`at`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)
- **`scaled_blas`**: [GroupedBlas.cpp_docs.md](./GroupedBlas.cpp_docs.md)


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
