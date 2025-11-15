# Keyword Index: `aten/src/ATen/native/EmbeddingBag.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/EmbeddingBag.cpp](../../../../../aten/src/ATen/native/EmbeddingBag.cpp)
- **Documentation**: [`EmbeddingBag.cpp_docs.md`](./EmbeddingBag.cpp_docs.md)
- **Folder**: `aten/src/ATen/native`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_embedding_bag_backward`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_backward_symint`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_cpu_impl_out`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_cpu_out`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_dense_backward_cpu`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_dense_backward_cpu_max`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_dense_backward_cpu_sum_mean`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_per_sample_weights_backward_cpu`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_per_sample_weights_backward_cpu_template`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`_embedding_bag_sparse_backward_symint`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`apply_bag_size`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`apply_bag_size_backward`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`check_arguments`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`constexpr`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`embedding_bag_cpu_max_out`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`fbgemm_spmdm_report_error_`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`if`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`is_fast_path`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`is_fast_path_index_select`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`is_fast_path_index_select_scale`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`make_bag_size`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`make_bag_size_out`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`make_max_indices`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`make_max_indices_out`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`make_offset2bag`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`make_offset2bag_out`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/Functions.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/Parallel.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/TensorOperators.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/TensorSubclassLikeUtils.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/cpu/vec/vec.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/native/CPUBlas.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/native/EmbeddingBag.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/native/NonSymbolicBC.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_backward_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_dense_backward.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_dense_backward_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_forward_only.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_forward_only_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_per_sample_weights_backward_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_sparse_backward.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/_embedding_bag_sparse_backward_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/embedding_backward_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/embedding_bag_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/empty.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/max.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/ones_like.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/resize_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/zero_native.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`c10/util/Half.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`c10/util/irange.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`caffe2/perfkernels/embedding_lookup_idx.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`cstring`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`fbgemm/Fbgemm.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`fbgemm/FbgemmConvert.h`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`tuple`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`utility`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`vector`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)

### Namespaces

- **`at`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`template`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)
- **`void`**: [EmbeddingBag.cpp_docs.md](./EmbeddingBag.cpp_docs.md)


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
