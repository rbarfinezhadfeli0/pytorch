# Keyword Index: `aten/src/ATen/native/cuda/Normalization.cu`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/Normalization.cu](../../../../../../aten/src/ATen/native/cuda/Normalization.cu)
- **Documentation**: [`Normalization.cu_docs.md`](./Normalization.cu_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Impl`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)

### Functions

- **`batch_norm_backward_elemt_cuda`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_calc_invstd`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_choose_impl`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_elementwise`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_elementwise_backward_eval`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_elementwise_backward_train`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_elemt_cuda`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_mean_var`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_update_stats`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_update_stats_and_invert`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`batch_norm_use_channels_last_kernels`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`first_type`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`if`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`is_mixed_type`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)

### Includes

- **`ATen/Functions.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/NativeFunctions.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/cuda/detail/IndexUtils.cuh`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/detail/CUDAHooksInterface.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/native/Normalization.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/native/ReduceOps.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/native/Resize.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/native/TensorIterator.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/native/cuda/Loops.cuh`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/native/cuda/Normalization.cuh`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/native/cuda/Resize.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/_batch_norm_with_update_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/batch_norm_backward_elemt_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/batch_norm_backward_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/batch_norm_backward_reduce_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/batch_norm_elemt_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/batch_norm_gather_stats_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/batch_norm_gather_stats_with_counts_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/batch_norm_stats_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/batch_norm_update_stats_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/cudnn_batch_norm.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/cudnn_batch_norm_backward.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/empty_like.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/from_blob.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/miopen_batch_norm.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/miopen_batch_norm_backward.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/native_batch_norm_backward_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/native_batch_norm_native.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`ATen/ops/scalar_tensor.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)
- **`c10/cuda/CUDAMathCompat.h`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)

### Namespaces

- **`at`**: [Normalization.cu_docs.md](./Normalization.cu_docs.md)


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
