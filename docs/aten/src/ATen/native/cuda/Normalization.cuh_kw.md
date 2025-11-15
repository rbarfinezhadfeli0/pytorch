# Keyword Index: `aten/src/ATen/native/cuda/Normalization.cuh`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/Normalization.cuh](../../../../../../aten/src/ATen/native/cuda/Normalization.cuh)
- **Documentation**: [`Normalization.cuh_docs.md`](./Normalization.cuh_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Float2`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`GradOp`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`InvStd`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`PtrTraits`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`SumReduceOp`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`Var`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)

### Functions

- **`Float2`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_elemt_channels_last_cuda_template`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_elemt_channels_last_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_elemt_channels_last_kernel_impl`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_elemt_cuda_template`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_elemt_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_elemt_kernel_impl`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_reduce_channels_last_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_backward_reduce_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_collect_statistics_channels_last_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_collect_statistics_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_elemt_channels_last_cuda_template`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_elemt_cuda_template`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_reduce_statistics_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_stats_channels_last_cuda_template`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_stats_cuda_template`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_transform_input_channels_last_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`batch_norm_transform_input_kernel`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`combine`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`flexible_launch_configs`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`for`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`getMSB`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`getNumThreads`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`if`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`merge_block_vertical_backward`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`reduce`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`warp_shfl_down`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`welford_merge_block_vertical`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`welford_merge_element`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)

### Includes

- **`ATen/AccumulateType.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/Dispatch.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/Functions.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/ceil_div.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/core/Tensor.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/cuda/CUDAContext.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/cuda/DeviceUtils.cuh`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/native/cuda/DeviceSqrt.cuh`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/native/cuda/LaunchUtils.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/native/cuda/block_reduce.cuh`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/ops/empty.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/ops/empty_like.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`ATen/ops/zeros.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)
- **`c10/macros/Macros.h`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)

### Namespaces

- **`at`**: [Normalization.cuh_docs.md](./Normalization.cuh_docs.md)


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
