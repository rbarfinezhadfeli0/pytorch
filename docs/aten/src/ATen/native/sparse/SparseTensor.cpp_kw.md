# Keyword Index: `aten/src/ATen/native/sparse/SparseTensor.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/sparse/SparseTensor.cpp](../../../../../../aten/src/ATen/native/sparse/SparseTensor.cpp)
- **Documentation**: [`SparseTensor.cpp_docs.md`](./SparseTensor.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/sparse`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_coalesce_sparse_cpu`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`_indices_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`_is_same_size_as_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`_nnz_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`_pin_memory_sparse_coo`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`_sparse_coo_tensor_unsafe`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`_sparse_coo_tensor_unsafe_symint`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`_validate_sparse_coo_tensor_args`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`_values_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`clone_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`coalesce`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`dense_dim_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`empty_like_sparse_coo`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`empty_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`empty_sparse_symint`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`expand_values_if_needed`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`indices_default`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`indices_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`is_coalesced_default`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`is_coalesced_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`is_pinned_sparse_coo`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`new_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`new_with_dims_and_tensor_sparse_symint`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`new_with_dims_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`sparse_coo_tensor`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`sparse_dim_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`sparse_mask`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`sparse_mask_projection`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`values_default`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`values_sparse`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/Functions.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/InitialTensorOptions.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/Layout.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/NamedTensorUtils.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/Parallel.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/SparseCsrTensorUtils.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/SparseTensorImpl.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/native/CPUBlas.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/native/Copy.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/native/IndexingUtils.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/native/NonSymbolicBC.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/native/SparseTensorUtils.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/native/sparse/SparseStubs.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_coalesce.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_coalesce_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_coalesced_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_convert_indices_from_csr_to_coo.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_dimI_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_dimV_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_indices_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_nnz_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_pin_memory_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_sparse_coo_tensor_unsafe_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_sparse_coo_tensor_with_dims.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_sparse_coo_tensor_with_dims_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_sparse_mask_projection_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_validate_sparse_coo_tensor_args_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/_values_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/clone_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/coalesce_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/copy_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/copy_sparse_to_sparse.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/copy_sparse_to_sparse_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/dense_dim_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/empty.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/empty_like_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/empty_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/index_select.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/indices_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/is_coalesced_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/is_pinned_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/ones.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/resize_as_sparse.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/resize_as_sparse_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/sparse_coo_tensor.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/sparse_coo_tensor_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/sparse_dim_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/sparse_mask_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/sparse_resize_and_clear_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/sparse_resize_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/to_dense_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/to_sparse_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/unique_dim.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/values_native.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`ATen/ops/zeros_like.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`c10/util/irange.h`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)

### Namespaces

- **`Tensor`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)
- **`at`**: [SparseTensor.cpp_docs.md](./SparseTensor.cpp_docs.md)


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
