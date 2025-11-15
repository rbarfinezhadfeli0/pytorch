# Keyword Index: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`

## File Information

- **Original File**: [aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h](../../../../../../../../aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h)
- **Documentation**: [`kernel_forward.h_docs.md`](./kernel_forward.h_docs.md)
- **Folder**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AttentionKernel`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`MM0`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`MM1`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`Params`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ScalingCoefs`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`SharedStorageAfterMM0`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`SharedStorageEpilogueAtEnd`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`SharedStorageEpilogueInLoop`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`iterators`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`thread`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)

### Functions

- **`advance_to_block`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`atomicMaxFloat`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`attention_kernel`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`check_supported`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`for`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`getBlocksGrid`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`getThreadsGrid`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`getWarpsPerSmFw`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`if`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`iterative_softmax`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`lane_id`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`thread_id`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`warp_id`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)

### Includes

- **`ATen/cuda/PhiloxUtils.cuh`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_rescale_output.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/transform/tile_smem_loader.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`c10/util/Exception.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cinttypes`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cmath`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`curand_kernel.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/bfloat16.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/epilogue/threadblock/default_epilogue_simt.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/epilogue/threadblock/default_epilogue_tensor_op.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/fast_math.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/device/default_gemm_configuration.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/gemm.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/kernel/default_gemm.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma_core_simt.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma_core_sm70.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma_core_sm75.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma_core_sm80.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/gemm/threadblock/threadblock_swizzle.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/layout/matrix.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/layout/vector.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/matrix.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/matrix_shape.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/numeric_types.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/platform/platform.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/tensor_ref.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`cutlass/transform/threadblock/predicated_tile_iterator.h`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`vector`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)

### Namespaces

- **`PyTorchMemEffAttention`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`gemm_kernel_utils`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)
- **`template`**: [kernel_forward.h_docs.md](./kernel_forward.h_docs.md)


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
