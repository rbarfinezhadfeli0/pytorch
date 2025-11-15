# Keyword Index: `aten/src/ATen/native/LinearAlgebraUtils.h`

## File Information

- **Original File**: [aten/src/ATen/native/LinearAlgebraUtils.h](../../../../../aten/src/ATen/native/LinearAlgebraUtils.h)
- **Documentation**: [`LinearAlgebraUtils.h_docs.md`](./LinearAlgebraUtils.h_docs.md)
- **Folder**: `aten/src/ATen/native`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`BroadcastLinearIndices`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)

### Functions

- **`_get_epsilon`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`_move_to_end`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`batchCount`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`batch_iterator_with_broadcasting`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`batched_matrix_contiguous_strides`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`checkAllSameDim`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`checkFloatingOrComplex`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`checkInputsSolver`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`checkIsMatrix`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`checkLinalgCompatibleDtype`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`checkNotComplexTolerance`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`checkSameDevice`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`checkUplo`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`cloneBatchedColumnMajor`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`computeLRWorkDim`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`copyBatchedColumnMajor`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`get_linear_indices`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`if`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`is_blas_compatible_column_major_order`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`is_blas_compatible_row_major_order`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`is_row_or_column_contiguous`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`linalg_solve_is_vector_rhs`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`linearSolveCheckInputs`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`matrixStride`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`same_stride_to`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`squareCheckInputs`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`svd_uses_cusolver`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`to_transpose_type`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)

### Includes

- **`ATen/ExpandUtils.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/Functions.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/TensorUtils.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/core/Tensor.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/native/TensorIterator.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/native/TransposeType.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/ops/arange.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/ops/empty.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/ops/empty_like.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/ops/empty_strided.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`ATen/ops/zeros.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`c10/core/ScalarType.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`c10/util/Exception.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`c10/util/irange.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`c10/util/strides.h`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`cctype`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`cstring`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`limits`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`sstream`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)
- **`type_traits`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)

### Namespaces

- **`at`**: [LinearAlgebraUtils.h_docs.md](./LinearAlgebraUtils.h_docs.md)


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
