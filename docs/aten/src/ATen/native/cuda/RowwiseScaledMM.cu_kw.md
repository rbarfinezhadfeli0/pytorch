# Keyword Index: `aten/src/ATen/native/cuda/RowwiseScaledMM.cu`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/RowwiseScaledMM.cu](../../../../../../aten/src/ATen/native/cuda/RowwiseScaledMM.cu)
- **Documentation**: [`RowwiseScaledMM.cu_docs.md`](./RowwiseScaledMM.cu_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Schedule`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)

### Functions

- **`ceildiv`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`check_inputs`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`constexpr`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`dispatch_fp8_rowwise_kernel_on_bias_dtype`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`dispatch_fp8_rowwise_kernel_on_cluster_size_and_transpose`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`dispatch_fp8_rowwise_kernel_on_fast_accum`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`dispatch_fp8_rowwise_kernel_on_input_dtypes`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`dispatch_fp8_rowwise_kernel_on_sm`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`dispatch_fp8_rowwise_kernel_on_tile_size`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`dispatch_fp8_rowwise_kernel_sm89`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`f8f8bf16_rowwise`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`f8f8bf16_rowwise_impl`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`f8f8bf16_rowwise_impl_sm100_sm120`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`f8f8bf16_rowwise_impl_sm89`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`handle_transposition`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`if`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`round_up_to_nearest_multiple`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`ATen/core/Tensor.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`ATen/cuda/CUDAContext.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`ATen/cuda/nvrtc_stub/ATenNVRTC.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`ATen/native/cuda/cutlass_common.cuh`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`c10/macros/Macros.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cute/atom/mma_atom.hpp`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cute/tensor.hpp`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/core_io.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/cutlass.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/epilogue/collective/collective_builder.hpp`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/epilogue/threadblock/fusion/visitors.hpp`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/gemm/collective/collective_builder.hpp`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/gemm/device/gemm.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/gemm/device/gemm_universal.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/gemm/device/gemm_universal_adapter.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/gemm/dispatch_policy.hpp`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/gemm/kernel/default_gemm_universal_with_visitor.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/gemm/kernel/gemm_universal.hpp`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/half.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/numeric_types.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/trace.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/util/host_tensor.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/util/packed_stride.hpp`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)
- **`cutlass/version.h`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)

### Namespaces

- **`at`**: [RowwiseScaledMM.cu_docs.md](./RowwiseScaledMM.cu_docs.md)


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
