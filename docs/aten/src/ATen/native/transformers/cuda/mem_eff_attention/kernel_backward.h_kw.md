# Keyword Index: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h`

## File Information

- **Original File**: [aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h](../../../../../../../../aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h)
- **Documentation**: [`kernel_backward.h_docs.md`](./kernel_backward.h_docs.md)
- **Folder**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AtomicLock`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`AttentionBackwardKernel`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`GmemTile`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`GradQTempStorage`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`MatmulDOIVJ`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`MatmulGradK`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`MatmulGradQ`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`MatmulGradV`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`MatmulQK`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`OutputFragments`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`Params`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`SharedStorageNoPrologue`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`SharedStoragePrologue`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`loads`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)

### Functions

- **`accumulateInGmem`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`acquire`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`advance_to_block`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`attention_kernel`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`check_supported`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`clear`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`computeDelta`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`for`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`gK_strideM`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`gQ_strideM`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`gV_strideM`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`getBlocksGrid`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`getNumParallelBlocksForQuery`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`getQueryEnd`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`getQueryStart`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`getQueryStartShift`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`getSmallestQueryForKey`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`getThreadsGrid`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`getWarpsPerSmBw`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`if`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`incrIteration`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`load`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`num_splits_key_device`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`o_strideM`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`print_size`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`processBlockIJ`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`prologueQkNextIteration`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`release`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`should_zero_workspace`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`split_key_device`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`store`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`storeAtomicAdd`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`workspace_elements_gk`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`workspace_elements_gq`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`workspace_elements_gv`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`workspace_size`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`workspace_strideBH`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`writeFragsToGmem`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`zfillGradKV`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)

### Includes

- **`ATen/cuda/PhiloxUtils.cuh`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_accum_lambda_iterator.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/iterators/epilogue_predicated_tile_iterator.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/transform/tile_smem_loader.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`c10/util/Exception.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cinttypes`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cmath`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cuda_fp16.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`curand_kernel.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/cutlass.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/epilogue/thread/linear_combination.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/epilogue/thread/linear_combination_relu.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/epilogue/thread/scale_type.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/epilogue/threadblock/epilogue_smem_accumulator.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/epilogue/warp/fragment_iterator_tensor_op.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/epilogue/warp/tile_iterator_tensor_op.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/fast_math.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/functional.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/gemm/device/default_gemm_configuration.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/gemm/gemm.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/gemm/kernel/default_gemm.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma_core_simt.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma_core_sm70.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma_core_sm75.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/gemm/threadblock/default_mma_core_sm80.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/integer_subbyte.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/layout/matrix.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/layout/vector.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/matrix_shape.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/numeric_conversion.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/numeric_types.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/platform/platform.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/tensor_ref.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/transform/threadblock/predicated_tile_iterator.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`cutlass/transform/threadblock/vector_iterator.h`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`type_traits`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`vector`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)

### Namespaces

- **`PyTorchMemEffAttention`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`gemm_kernel_utils`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)
- **`template`**: [kernel_backward.h_docs.md](./kernel_backward.h_docs.md)


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
