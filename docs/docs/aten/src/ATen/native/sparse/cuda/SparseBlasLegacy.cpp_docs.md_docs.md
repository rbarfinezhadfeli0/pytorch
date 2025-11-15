# Documentation: `docs/aten/src/ATen/native/sparse/cuda/SparseBlasLegacy.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/sparse/cuda/SparseBlasLegacy.cpp_docs.md`
- **Size**: 5,261 bytes (5.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/sparse/cuda/SparseBlasLegacy.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/sparse/cuda/SparseBlasLegacy.cpp`
- **Size**: 2,544 bytes (2.48 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
/*
Functions here use deprecated cuSPARSE API that was removed in CUDA 11.
This file will be removed eventually.
*/
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/sparse/cuda/SparseBlasLegacy.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>

namespace at::native {

void s_addmm_out_csr_sparse_dense_cuda_worker(int64_t nnz, int64_t m, int64_t n, int64_t k, const Tensor& r_, const Scalar& beta, const Tensor& t, const Scalar& alpha, const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, const Tensor& dense) {
  TORCH_INTERNAL_ASSERT(nnz > 0);

  // No half support, so we don't have to use CUDATypeConversion
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(), "addmm_sparse_cuda", [&] {
        scalar_t cast_beta = beta.to<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();
        Tensor r__;
        if (cast_beta == scalar_t(0)) {
          r_.zero_();
        } else if (!at::sparse::is_same_tensor(t, r_)) {
          r_.copy_(t);
        }
        if (r_.stride(0) == 1 && r_.stride(1) == r_.size(0)) {
          r__ = r_;
        } else {
          // Note: This storage arrangement is preferred due to most of the CUDA kernels handle only contiguous tensors
          r__ = r_.transpose(0, 1).clone(at::MemoryFormat::Contiguous);
          r__.transpose_(0, 1);
        }
        TORCH_INTERNAL_ASSERT(r__.mT().is_contiguous());
        Tensor dense_;
        char transpose_dense;
        if (dense.stride(0) == 1 && dense.stride(1) == dense.size(0)) {
          transpose_dense = 'n';
          dense_ = dense;
        } else if (dense.stride(1) == 1 && dense.stride(0) == dense.size(1)) {
          transpose_dense = 't';
          dense_ = dense;
        } else {
          transpose_dense = 't';
          dense_ = dense.contiguous();
        }

        sparse::cuda::csrmm2(
          'n',
          transpose_dense,
          m,
          n,
          k,
          nnz,
          cast_alpha,
          values.data_ptr<scalar_t>(),
          crow_indices.data_ptr<int32_t>(),
          col_indices.data_ptr<int32_t>(),
          dense_.data_ptr<scalar_t>(),
          (transpose_dense == 'n' ? dense_.stride(1) : dense_.stride(0)),
          cast_beta,
          r__.data_ptr<scalar_t>(),
          r__.stride(1));

        if (!at::sparse::is_same_tensor(r__, r_)) {
          r_.copy_(r__);
        }
      }
    );
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/sparse/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/native/SparseTensorUtils.h`
- `ATen/native/sparse/cuda/SparseBlasLegacy.h`
- `ATen/native/sparse/cuda/SparseCUDABlas.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`aten/src/ATen/native/sparse/cuda`):

- [`cuSPARSELtOps.cpp_docs.md`](./cuSPARSELtOps.cpp_docs.md)
- [`SparseCsrTensorMath.cu_docs.md`](./SparseCsrTensorMath.cu_docs.md)
- [`SparseSemiStructuredOps.cu_docs.md`](./SparseSemiStructuredOps.cu_docs.md)
- [`SparseCUDABlas.h_docs.md`](./SparseCUDABlas.h_docs.md)
- [`SparseMatMul.cu_docs.md`](./SparseMatMul.cu_docs.md)
- [`SparseCUDATensorMath.cuh_docs.md`](./SparseCUDATensorMath.cuh_docs.md)
- [`cuSPARSELtOps.h_docs.md`](./cuSPARSELtOps.h_docs.md)
- [`SparseBlas.cpp_docs.md`](./SparseBlas.cpp_docs.md)
- [`StaticSort.h_docs.md`](./StaticSort.h_docs.md)
- [`SparseSemiStructuredApplyDense.cu_docs.md`](./SparseSemiStructuredApplyDense.cu_docs.md)


## Cross-References

- **File Documentation**: `SparseBlasLegacy.cpp_docs.md`
- **Keyword Index**: `SparseBlasLegacy.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/sparse/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/sparse/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/aten/src/ATen/native/sparse/cuda`):

- [`SparseBlasLegacy.h_docs.md_docs.md`](./SparseBlasLegacy.h_docs.md_docs.md)
- [`SparseBlasImpl.h_kw.md_docs.md`](./SparseBlasImpl.h_kw.md_docs.md)
- [`SparseBlasLegacy.h_kw.md_docs.md`](./SparseBlasLegacy.h_kw.md_docs.md)
- [`SparseMatMul.cu_docs.md_docs.md`](./SparseMatMul.cu_docs.md_docs.md)
- [`SparseCUDABlas.cpp_kw.md_docs.md`](./SparseCUDABlas.cpp_kw.md_docs.md)
- [`SparseCUDATensorMath.cu_kw.md_docs.md`](./SparseCUDATensorMath.cu_kw.md_docs.md)
- [`cuSPARSELtOps.cpp_kw.md_docs.md`](./cuSPARSELtOps.cpp_kw.md_docs.md)
- [`SparseBlasLegacy.cpp_kw.md_docs.md`](./SparseBlasLegacy.cpp_kw.md_docs.md)
- [`SoftMax.cu_kw.md_docs.md`](./SoftMax.cu_kw.md_docs.md)


## Cross-References

- **File Documentation**: `SparseBlasLegacy.cpp_docs.md_docs.md`
- **Keyword Index**: `SparseBlasLegacy.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
