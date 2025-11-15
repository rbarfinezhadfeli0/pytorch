# Keyword Index: `aten/src/ATen/native/Normalization.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/Normalization.cpp](../../../../../aten/src/ATen/native/Normalization.cpp)
- **Documentation**: [`Normalization.cpp_docs.md`](./Normalization.cpp_docs.md)
- **Folder**: `aten/src/ATen/native`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`InvStd`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`Var`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`VarTransform`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)

### Functions

- **`_select_batch_norm_backend`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`batch_norm`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`check_dims_match_num_input_features`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`if`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`instance_norm`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`is_contiguous`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`repeat_if_defined`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`suggest_memory_format_contig`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)

### Includes

- **`ATen/AccumulateType.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/Config.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/Dispatch.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/Functions.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/OpMathType.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/Parallel.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ScalarOps.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/TensorIterator.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/TensorMeta.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/TensorOperators.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/detail/CUDAHooksInterface.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/native/Normalization.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/native/Resize.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/native/batch_norm.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/native/cpu/Loops.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/native/cpu/mixed_data_type.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_batch_norm_impl_index.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_batch_norm_impl_index_backward_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_batch_norm_impl_index_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_batch_norm_no_update.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_batch_norm_no_update_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_batch_norm_with_update.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_batch_norm_with_update_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_native_batch_norm_legit.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_native_batch_norm_legit_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_native_batch_norm_legit_no_training.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/_native_batch_norm_legit_no_training_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/alias.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/batch_norm.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/batch_norm_backward_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/batch_norm_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/batch_norm_update_stats_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/cudnn_batch_norm.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/cudnn_batch_norm_backward.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/empty.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/empty_like.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/instance_norm_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/linalg_vector_norm.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/mean.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/miopen_batch_norm.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/miopen_batch_norm_backward.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/mul.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/native_batch_norm.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/native_batch_norm_backward.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/native_batch_norm_backward_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/native_batch_norm_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/renorm_native.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/sqrt.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`ATen/ops/sum.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`c10/core/SymIntArrayRef.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`c10/util/irange.h`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`utility`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)
- **`vector`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)

### Namespaces

- **`at`**: [Normalization.cpp_docs.md](./Normalization.cpp_docs.md)


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
