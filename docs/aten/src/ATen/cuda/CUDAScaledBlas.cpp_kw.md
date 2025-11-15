# Keyword Index: `aten/src/ATen/cuda/CUDAScaledBlas.cpp`

## File Information

- **Original File**: [aten/src/ATen/cuda/CUDAScaledBlas.cpp](../../../../../aten/src/ATen/cuda/CUDAScaledBlas.cpp)
- **Documentation**: [`CUDAScaledBlas.cpp_docs.md`](./CUDAScaledBlas.cpp_docs.md)
- **Folder**: `aten/src/ATen/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`check_deepseek_recipe`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`check_mxfp4_recipe`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`check_mxfp8_recipe`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`check_nvfp4_recipe`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`check_nvfp4_recipe_single_scale`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`check_rowwise_recipe`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`check_tensorwise_recipe`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`if`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ExpandUtils.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/Functions.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/OpMathType.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ceil_div.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/core/NamedTensor.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/cuda/CUDABlas.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/cuda/tunable/Tunable.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/cuda/tunable/TunableGemm.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/native/GroupedMMUtils.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/native/Resize.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/native/cuda/GroupMM.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/native/cuda/RowwiseScaledMM.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/native/cuda/ScaledGroupMM.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/_addmm_activation_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/_efficientzerotensor.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/_scaled_mm_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/_unsafe_view_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/abs.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/addmm_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/addmv_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/baddbmm_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/bmm_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/copy_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/dot_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/empty.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/empty_strided.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/gelu.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/max.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/mm_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/mul.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/ones.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/relu.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/scalar_tensor_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`ATen/ops/vdot_native.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`c10/core/Scalar.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`c10/core/ScalarType.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`c10/util/Exception.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`c10/util/MaybeOwned.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`c10/util/SmallVector.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`c10/util/typeid.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`cstdint`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)
- **`fbgemm_gpu/torch_ops.h`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)

### Namespaces

- **`at`**: [CUDAScaledBlas.cpp_docs.md](./CUDAScaledBlas.cpp_docs.md)


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
