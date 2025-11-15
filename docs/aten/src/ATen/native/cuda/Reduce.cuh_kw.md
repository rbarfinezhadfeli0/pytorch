# Keyword Index: `aten/src/ATen/native/cuda/Reduce.cuh`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/Reduce.cuh](../../../../../../aten/src/ATen/native/cuda/Reduce.cuh)
- **Documentation**: [`Reduce.cuh_docs.md`](./Reduce.cuh_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AccumulationBuffer`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ReduceConfig`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ReduceJitOp`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ReduceOp`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`T`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`T1`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`T2`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`func_wrapper_t`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`mnt_wrapper`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)

### Functions

- **`block`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`div_up`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`for`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`get_accumulated_output`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`get_output_vec_size`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`global_memory_size`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`gpu_reduce_kernel`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`grid`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`if`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`input_idx`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`input_vectorized_thread_reduce_impl`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`jitted_gpu_reduce_kernel`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`last_pow2`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`launch_jitted_reduce_kernel`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`launch_reduce_kernel`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`mark_block_finished`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`max_reduce_threads`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`mock_values_per_thread`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`output_idx`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`project`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`reduce`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`reduce_fraction`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`reduce_kernel`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`run`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`semaphore_size`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`setReduceConfig`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`set_block_dimension`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`set_results`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`set_results_to_output`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`shared_memory_offset`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`shared_memory_size`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`should_block_x_reduce`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`should_block_y_reduce`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`should_global_reduce`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`should_reduce_tail`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`should_store`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`split_input`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`split_output`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`staging_memory_offset`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`translate_idx`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`values_per_thread`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`warp_shfl_down`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)

### Includes

- **`ATen/OpMathType.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/cuda/CUDAContext.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/cuda/DeviceUtils.cuh`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/cuda/detail/OffsetCalculator.cuh`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/detail/FunctionTraits.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/native/TensorIterator.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/native/cuda/KernelUtils.cuh`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/native/cuda/MemoryAccess.cuh`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/native/cuda/jit_utils.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`ATen/native/cuda/thread_constants.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`array`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`c10/cuda/CUDACachingAllocator.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`c10/macros/Macros.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`functional`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`iosfwd`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`thrust/pair.h`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`type_traits`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)
- **`utility`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)

### Namespaces

- **`at`**: [Reduce.cuh_docs.md](./Reduce.cuh_docs.md)


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
