# Documentation: `aten/src/ATen/mkl/SparseBlas.h`

## File Metadata

- **Path**: `aten/src/ATen/mkl/SparseBlas.h`
- **Size**: 8,067 bytes (7.88 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

/*
  Provides a subset of MKL Sparse BLAS functions as templates:

    mv<scalar_t>(operation, alpha, A, descr, x, beta, y)

  where scalar_t is double, float, c10::complex<double> or c10::complex<float>.
  The functions are available in at::mkl::sparse namespace.
*/

#include <c10/util/Exception.h>
#include <c10/util/complex.h>

#include <mkl_spblas.h>

namespace at::mkl::sparse {

#define MKL_SPARSE_CREATE_CSR_ARGTYPES(scalar_t)                              \
  sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, \
      const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,             \
      MKL_INT *col_indx, scalar_t *values

template <typename scalar_t>
inline void create_csr(MKL_SPARSE_CREATE_CSR_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::create_csr: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void create_csr<float>(MKL_SPARSE_CREATE_CSR_ARGTYPES(float));
template <>
void create_csr<double>(MKL_SPARSE_CREATE_CSR_ARGTYPES(double));
template <>
void create_csr<c10::complex<float>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<float>));
template <>
void create_csr<c10::complex<double>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_CREATE_BSR_ARGTYPES(scalar_t)                   \
  sparse_matrix_t *A, const sparse_index_base_t indexing,          \
      const sparse_layout_t block_layout, const MKL_INT rows,      \
      const MKL_INT cols, MKL_INT block_size, MKL_INT *rows_start, \
      MKL_INT *rows_end, MKL_INT *col_indx, scalar_t *values

template <typename scalar_t>
inline void create_bsr(MKL_SPARSE_CREATE_BSR_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::create_bsr: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void create_bsr<float>(MKL_SPARSE_CREATE_BSR_ARGTYPES(float));
template <>
void create_bsr<double>(MKL_SPARSE_CREATE_BSR_ARGTYPES(double));
template <>
void create_bsr<c10::complex<float>>(
    MKL_SPARSE_CREATE_BSR_ARGTYPES(c10::complex<float>));
template <>
void create_bsr<c10::complex<double>>(
    MKL_SPARSE_CREATE_BSR_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_MV_ARGTYPES(scalar_t)                        \
  const sparse_operation_t operation, const scalar_t alpha,     \
      const sparse_matrix_t A, const struct matrix_descr descr, \
      const scalar_t *x, const scalar_t beta, scalar_t *y

template <typename scalar_t>
inline void mv(MKL_SPARSE_MV_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::mv: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void mv<float>(MKL_SPARSE_MV_ARGTYPES(float));
template <>
void mv<double>(MKL_SPARSE_MV_ARGTYPES(double));
template <>
void mv<c10::complex<float>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<float>));
template <>
void mv<c10::complex<double>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_ADD_ARGTYPES(scalar_t)                      \
  const sparse_operation_t operation, const sparse_matrix_t A, \
      const scalar_t alpha, const sparse_matrix_t B, sparse_matrix_t *C

template <typename scalar_t>
inline void add(MKL_SPARSE_ADD_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::add: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void add<float>(MKL_SPARSE_ADD_ARGTYPES(float));
template <>
void add<double>(MKL_SPARSE_ADD_ARGTYPES(double));
template <>
void add<c10::complex<float>>(MKL_SPARSE_ADD_ARGTYPES(c10::complex<float>));
template <>
void add<c10::complex<double>>(MKL_SPARSE_ADD_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_EXPORT_CSR_ARGTYPES(scalar_t)                              \
  const sparse_matrix_t source, sparse_index_base_t *indexing, MKL_INT *rows, \
      MKL_INT *cols, MKL_INT **rows_start, MKL_INT **rows_end,                \
      MKL_INT **col_indx, scalar_t **values

template <typename scalar_t>
inline void export_csr(MKL_SPARSE_EXPORT_CSR_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::export_csr: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void export_csr<float>(MKL_SPARSE_EXPORT_CSR_ARGTYPES(float));
template <>
void export_csr<double>(MKL_SPARSE_EXPORT_CSR_ARGTYPES(double));
template <>
void export_csr<c10::complex<float>>(
    MKL_SPARSE_EXPORT_CSR_ARGTYPES(c10::complex<float>));
template <>
void export_csr<c10::complex<double>>(
    MKL_SPARSE_EXPORT_CSR_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_MM_ARGTYPES(scalar_t)                                      \
  const sparse_operation_t operation, const scalar_t alpha,                   \
      const sparse_matrix_t A, const struct matrix_descr descr,               \
      const sparse_layout_t layout, const scalar_t *B, const MKL_INT columns, \
      const MKL_INT ldb, const scalar_t beta, scalar_t *C, const MKL_INT ldc

template <typename scalar_t>
inline void mm(MKL_SPARSE_MM_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::mm: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void mm<float>(MKL_SPARSE_MM_ARGTYPES(float));
template <>
void mm<double>(MKL_SPARSE_MM_ARGTYPES(double));
template <>
void mm<c10::complex<float>>(MKL_SPARSE_MM_ARGTYPES(c10::complex<float>));
template <>
void mm<c10::complex<double>>(MKL_SPARSE_MM_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_SPMMD_ARGTYPES(scalar_t)                               \
  const sparse_operation_t operation, const sparse_matrix_t A,            \
      const sparse_matrix_t B, const sparse_layout_t layout, scalar_t *C, \
      const MKL_INT ldc

template <typename scalar_t>
inline void spmmd(MKL_SPARSE_SPMMD_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::spmmd: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void spmmd<float>(MKL_SPARSE_SPMMD_ARGTYPES(float));
template <>
void spmmd<double>(MKL_SPARSE_SPMMD_ARGTYPES(double));
template <>
void spmmd<c10::complex<float>>(MKL_SPARSE_SPMMD_ARGTYPES(c10::complex<float>));
template <>
void spmmd<c10::complex<double>>(
    MKL_SPARSE_SPMMD_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_TRSV_ARGTYPES(scalar_t)                      \
  const sparse_operation_t operation, const scalar_t alpha,     \
      const sparse_matrix_t A, const struct matrix_descr descr, \
      const scalar_t *x, scalar_t *y

template <typename scalar_t>
inline sparse_status_t trsv(MKL_SPARSE_TRSV_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::trsv: not implemented for ",
      typeid(scalar_t).name());
}

template <>
sparse_status_t trsv<float>(MKL_SPARSE_TRSV_ARGTYPES(float));
template <>
sparse_status_t trsv<double>(MKL_SPARSE_TRSV_ARGTYPES(double));
template <>
sparse_status_t trsv<c10::complex<float>>(MKL_SPARSE_TRSV_ARGTYPES(c10::complex<float>));
template <>
sparse_status_t trsv<c10::complex<double>>(MKL_SPARSE_TRSV_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_TRSM_ARGTYPES(scalar_t)                                    \
  const sparse_operation_t operation, const scalar_t alpha,                   \
      const sparse_matrix_t A, const struct matrix_descr descr,               \
      const sparse_layout_t layout, const scalar_t *x, const MKL_INT columns, \
      const MKL_INT ldx, scalar_t *y, const MKL_INT ldy

template <typename scalar_t>
inline sparse_status_t trsm(MKL_SPARSE_TRSM_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::trsm: not implemented for ",
      typeid(scalar_t).name());
}

template <>
sparse_status_t trsm<float>(MKL_SPARSE_TRSM_ARGTYPES(float));
template <>
sparse_status_t trsm<double>(MKL_SPARSE_TRSM_ARGTYPES(double));
template <>
sparse_status_t trsm<c10::complex<float>>(MKL_SPARSE_TRSM_ARGTYPES(c10::complex<float>));
template <>
sparse_status_t trsm<c10::complex<double>>(MKL_SPARSE_TRSM_ARGTYPES(c10::complex<double>));

} // namespace at::mkl::sparse

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `matrix_descr`, `matrix_descr`, `matrix_descr`, `matrix_descr`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/mkl`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `c10/util/complex.h`
- `mkl_spblas.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/mkl`):

- [`Exceptions.h_docs.md`](./Exceptions.h_docs.md)
- [`Limits.h_docs.md`](./Limits.h_docs.md)
- [`SparseBlas.cpp_docs.md`](./SparseBlas.cpp_docs.md)
- [`Sparse.h_docs.md`](./Sparse.h_docs.md)
- [`Descriptors.h_docs.md`](./Descriptors.h_docs.md)
- [`Utils.h_docs.md`](./Utils.h_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`SparseDescriptors.h_docs.md`](./SparseDescriptors.h_docs.md)


## Cross-References

- **File Documentation**: `SparseBlas.h_docs.md`
- **Keyword Index**: `SparseBlas.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
