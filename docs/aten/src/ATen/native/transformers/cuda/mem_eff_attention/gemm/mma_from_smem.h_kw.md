# Keyword Index: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h`

## File Information

- **Original File**: [aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h](../../../../../../../../../aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h)
- **Documentation**: [`mma_from_smem.h_docs.md`](./mma_from_smem.h_docs.md)
- **Folder**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AccumulatorSharedStorage`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`B2bGemm`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`DefaultMmaFromSharedMemory`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`Detail`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`FragmentElementwiseScaler`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`MmaBaseFromSharedMemory`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`MmaMultistageFromSharedMemory`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`MmaPipelinedFromSharedMemory`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`NoOpWarpIteratorScale`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`SharedStorage`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`from`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`using`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)

### Functions

- **`LayoutAccum`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`LayoutB`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`NoOpWarpIteratorScale`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`_prologue`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`accumApplyLSEToSmem`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`accumToSmem`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`accum_ref`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`apply`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`copy_tiles_and_advance_1`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`for`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`if`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`load`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`operand_B_ref`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`prologue`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`set_prologue_done`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)

### Includes

- **`ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_thread_apply_logsumexp.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_accum_lambda_iterator.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/iterators/default_warp_iterator_from_smem.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/iterators/make_residual_last.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/iterators/transpose_warp_iterator.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/iterators/warp_iterator_from_smem.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/aligned_buffer.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/arch/memory.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/array.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/cutlass.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/epilogue/thread/linear_combination.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/epilogue/threadblock/default_epilogue_simt.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/epilogue/threadblock/default_epilogue_tensor_op.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/epilogue/threadblock/epilogue_smem_accumulator.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/functional.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/gemm/gemm.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/gemm/threadblock/mma_base.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/gemm/threadblock/mma_multistage.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/gemm/threadblock/mma_pipelined.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/gemm/warp/mma_tensor_op_tile_access_iterator.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/matrix_shape.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/numeric_conversion.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/numeric_types.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/platform/platform.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`cutlass/transform/threadblock/vector_iterator.h`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)

### Namespaces

- **`cutlass`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`gemm`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)
- **`threadblock`**: [mma_from_smem.h_docs.md](./mma_from_smem.h_docs.md)


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
