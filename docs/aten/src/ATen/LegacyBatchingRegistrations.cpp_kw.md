# Keyword Index: `aten/src/ATen/LegacyBatchingRegistrations.cpp`

## File Information

- **Original File**: [aten/src/ATen/LegacyBatchingRegistrations.cpp](../../../../aten/src/ATen/LegacyBatchingRegistrations.cpp)
- **Documentation**: [`LegacyBatchingRegistrations.cpp_docs.md`](./LegacyBatchingRegistrations.cpp_docs.md)
- **Folder**: `aten/src/ATen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`a`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)

### Functions

- **`_has_same_storage_numel_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`_make_dual_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`_new_zeros_with_same_feature_meta_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`_reshape_alias_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`as_strided_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`binary_pointwise_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`bmm_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`cat_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`checkBasicAsStridedValidForSlice`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`checkBatchDimsAtFrontInLayout`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`clamp_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`clamp_max_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`clamp_min_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`clone_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`comparison_pointwise_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`contiguous_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`diagonal_backward_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`diagonal_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`dot_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`expand_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`getGradInputPhysicalDim`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`if`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`isPhysicalScalarTensor`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`is_allowed_dim_on_scalar_tensor`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`mm_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`movedim_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`mv_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`new_empty_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`new_empty_strided_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`new_zeros_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`permute_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`pow_scalar_Tensor_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`reshape_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`select_backward_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`select_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`slice_backward_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`slice_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`squeeze_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`squeeze_dim_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`squeeze_dims_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`stack_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`sum_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`to_dtype_layout_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`trace_backward_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`trace_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`transpose_int_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`unfold_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`unsqueeze_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`unwrap_and_call`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`unwrap_and_call_method`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`view_as_complex_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`view_batching_rule`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)

### Includes

- **`ATen/ATen.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`ATen/LegacyBatchedFallback.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`ATen/LegacyVmapTransforms.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`ATen/RedispatchFunctions.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`ATen/core/IListRef.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`ATen/native/ResizeCommon.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`c10/core/SymIntArrayRef.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`c10/util/irange.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`torch/library.h`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)
- **`utility`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)

### Namespaces

- **`at`**: [LegacyBatchingRegistrations.cpp_docs.md](./LegacyBatchingRegistrations.cpp_docs.md)


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
