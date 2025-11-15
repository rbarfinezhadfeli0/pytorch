# Keyword Index: `aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu`

## File Information

- **Original File**: [aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu](../../../../../../../aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu)
- **Documentation**: [`SparseCUDATensorMath.cu_docs.md`](./SparseCUDATensorMath.cu_docs.md)
- **Folder**: `aten/src/ATen/native/sparse/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`TensorCAddOp`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`TensorMulOp`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)

### Functions

- **`_sparse_sum_backward_cuda`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`_sparse_sum_backward_cuda_kernel`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`_to_csr_int`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`addmm_sparse_dense_cuda`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`bmm_sparse_cuda`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`getTensorCudaDataType`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`hspmm_sparse_cuda`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`s_addmm_out_sparse_dense_cuda_worker`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`s_addmm_sparse_dense_cuda`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`search_end_matrix_indices`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`search_end_matrix_indices_cuda_kernel`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ExpandUtils.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/Functions.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/NativeFunctions.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/SparseCsrTensorUtils.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/WrapDimUtilsMulti.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/core/Tensor.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/cuda/CUDAContext.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/cuda/CUDAUtils.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/cuda/ThrustAllocator.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/cuda/detail/IndexUtils.cuh`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/native/SparseTensorUtils.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/native/sparse/SparseTensorMath.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/native/sparse/cuda/SparseBlasLegacy.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/native/sparse/cuda/SparseCUDABlas.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/native/sparse/cuda/SparseCUDATensorMath.cuh`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/_sparse_sum_native.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/add_native.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/addmm_native.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/bmm_native.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/cat.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/copy_sparse_to_sparse.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/empty.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/empty_like.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/hspmm_native.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/mul.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/result_type.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/scalar_tensor.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`ATen/ops/zeros_like.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`bitset`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`c10/cuda/CUDACachingAllocator.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`cuda_runtime_api.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`cusparse.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`memory`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`thrust/binary_search.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`thrust/device_ptr.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`thrust/sequence.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`thrust/sort.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)
- **`thrust/system/cuda/execution_policy.h`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)

### Namespaces

- **`at`**: [SparseCUDATensorMath.cu_docs.md](./SparseCUDATensorMath.cu_docs.md)


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
