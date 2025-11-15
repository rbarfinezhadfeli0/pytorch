# Documentation: `docs/aten/src/ATen/cuda/CUDASparseDescriptors.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/CUDASparseDescriptors.h_docs.md`
- **Size**: 10,689 bytes (10.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/CUDASparseDescriptors.h`

## File Metadata

- **Path**: `aten/src/ATen/cuda/CUDASparseDescriptors.h`
- **Size**: 7,769 bytes (7.59 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDASparse.h>

#include <c10/core/ScalarType.h>

#if defined(USE_ROCM)
#include <type_traits>
#endif

namespace at::cuda::sparse {

template <typename T, cusparseStatus_t (*destructor)(T*)>
struct CuSparseDescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDASPARSE_CHECK(destructor(x));
    }
  }
};

template <typename T, cusparseStatus_t (*destructor)(T*)>
class CuSparseDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, CuSparseDescriptorDeleter<T, destructor>> descriptor_;
};

template <typename T, cusparseStatus_t (*destructor)(const T*)>
struct ConstCuSparseDescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDASPARSE_CHECK(destructor(x));
    }
  }
};

template <typename T, cusparseStatus_t (*destructor)(const T*)>
class ConstCuSparseDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, ConstCuSparseDescriptorDeleter<T, destructor>> descriptor_;
};

#if defined(USE_ROCM)
using cusparseMatDescr = std::remove_pointer_t<hipsparseMatDescr_t>;
using cusparseDnMatDescr = std::remove_pointer_t<hipsparseDnMatDescr_t>;
using cusparseDnVecDescr = std::remove_pointer_t<hipsparseDnVecDescr_t>;
using cusparseSpMatDescr = std::remove_pointer_t<hipsparseSpMatDescr_t>;
using cusparseSpMatDescr = std::remove_pointer_t<hipsparseSpMatDescr_t>;
using cusparseSpGEMMDescr = std::remove_pointer_t<hipsparseSpGEMMDescr_t>;
#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()
using bsrsv2Info = std::remove_pointer_t<bsrsv2Info_t>;
using bsrsm2Info = std::remove_pointer_t<bsrsm2Info_t>;
#endif
#endif

// NOTE: This is only needed for CUDA 11 and earlier, since CUDA 12 introduced
// API for const descriptors
cusparseStatus_t destroyConstDnMat(const cusparseDnMatDescr* dnMatDescr);

class TORCH_CUDA_CPP_API CuSparseMatDescriptor
    : public CuSparseDescriptor<cusparseMatDescr, &cusparseDestroyMatDescr> {
 public:
  CuSparseMatDescriptor() {
    cusparseMatDescr_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }

  CuSparseMatDescriptor(bool upper, bool unit) {
    cusparseFillMode_t fill_mode =
        upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_type =
        unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseMatDescr_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&raw_descriptor));
    TORCH_CUDASPARSE_CHECK(cusparseSetMatFillMode(raw_descriptor, fill_mode));
    TORCH_CUDASPARSE_CHECK(cusparseSetMatDiagType(raw_descriptor, diag_type));
    descriptor_.reset(raw_descriptor);
  }
};

#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()

class TORCH_CUDA_CPP_API CuSparseBsrsv2Info
    : public CuSparseDescriptor<bsrsv2Info, &cusparseDestroyBsrsv2Info> {
 public:
  CuSparseBsrsv2Info() {
    bsrsv2Info_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseCreateBsrsv2Info(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

class TORCH_CUDA_CPP_API CuSparseBsrsm2Info
    : public CuSparseDescriptor<bsrsm2Info, &cusparseDestroyBsrsm2Info> {
 public:
  CuSparseBsrsm2Info() {
    bsrsm2Info_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseCreateBsrsm2Info(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

#endif // AT_USE_HIPSPARSE_TRIANGULAR_SOLVE

cusparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type);

  class TORCH_CUDA_CPP_API CuSparseDnMatDescriptor
      : public ConstCuSparseDescriptor<
            cusparseDnMatDescr,
            &cusparseDestroyDnMat> {
   public:
    explicit CuSparseDnMatDescriptor(
        const Tensor& input,
        int64_t batch_offset = -1);
  };

  class TORCH_CUDA_CPP_API CuSparseConstDnMatDescriptor
      : public ConstCuSparseDescriptor<
            const cusparseDnMatDescr,
            &destroyConstDnMat> {
   public:
    explicit CuSparseConstDnMatDescriptor(
        const Tensor& input,
        int64_t batch_offset = -1);
  cusparseDnMatDescr* unsafe_mutable_descriptor() const {
    return const_cast<cusparseDnMatDescr*>(descriptor());
  }
  cusparseDnMatDescr* unsafe_mutable_descriptor() {
    return const_cast<cusparseDnMatDescr*>(descriptor());
  }
  };

  class TORCH_CUDA_CPP_API CuSparseDnVecDescriptor
      : public ConstCuSparseDescriptor<
            cusparseDnVecDescr,
            &cusparseDestroyDnVec> {
   public:
    explicit CuSparseDnVecDescriptor(const Tensor& input);
  };

  class TORCH_CUDA_CPP_API CuSparseSpMatDescriptor
      : public ConstCuSparseDescriptor<
            cusparseSpMatDescr,
            &cusparseDestroySpMat> {};

class TORCH_CUDA_CPP_API CuSparseSpMatCsrDescriptor
    : public CuSparseSpMatDescriptor {
 public:
  explicit CuSparseSpMatCsrDescriptor(const Tensor& input, int64_t batch_offset = -1);

  std::tuple<int64_t, int64_t, int64_t> get_size() {
    int64_t rows = 0, cols = 0, nnz = 0;
    TORCH_CUDASPARSE_CHECK(cusparseSpMatGetSize(
        this->descriptor(),
        &rows,
        &cols,
        &nnz));
    return std::make_tuple(rows, cols, nnz);
  }

  void set_tensor(const Tensor& input) {
    auto crow_indices = input.crow_indices();
    auto col_indices = input.col_indices();
    auto values = input.values();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(crow_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(col_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
    TORCH_CUDASPARSE_CHECK(cusparseCsrSetPointers(
        this->descriptor(),
        crow_indices.data_ptr(),
        col_indices.data_ptr(),
        values.data_ptr()));
  }

#if AT_USE_CUSPARSE_GENERIC_SPSV()
  void set_mat_fill_mode(bool upper) {
    cusparseFillMode_t fill_mode =
        upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER;
    TORCH_CUDASPARSE_CHECK(cusparseSpMatSetAttribute(
        this->descriptor(),
        CUSPARSE_SPMAT_FILL_MODE,
        &fill_mode,
        sizeof(fill_mode)));
  }

  void set_mat_diag_type(bool unit) {
    cusparseDiagType_t diag_type =
        unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    TORCH_CUDASPARSE_CHECK(cusparseSpMatSetAttribute(
        this->descriptor(),
        CUSPARSE_SPMAT_DIAG_TYPE,
        &diag_type,
        sizeof(diag_type)));
  }
#endif
};

#if AT_USE_CUSPARSE_GENERIC_SPSV()
class TORCH_CUDA_CPP_API CuSparseSpSVDescriptor
    : public CuSparseDescriptor<cusparseSpSVDescr, &cusparseSpSV_destroyDescr> {
 public:
  CuSparseSpSVDescriptor() {
    cusparseSpSVDescr_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseSpSV_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

#if AT_USE_CUSPARSE_GENERIC_SPSM()
class TORCH_CUDA_CPP_API CuSparseSpSMDescriptor
    : public CuSparseDescriptor<cusparseSpSMDescr, &cusparseSpSM_destroyDescr> {
 public:
  CuSparseSpSMDescriptor() {
    cusparseSpSMDescr_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseSpSM_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

class TORCH_CUDA_CPP_API CuSparseSpGEMMDescriptor
    : public CuSparseDescriptor<cusparseSpGEMMDescr, &cusparseSpGEMM_destroyDescr> {
 public:
  CuSparseSpGEMMDescriptor() {
    cusparseSpGEMMDescr_t raw_descriptor = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

} // namespace at::cuda::sparse

```



## High-Level Overview


This C++ file contains approximately 13 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `CuSparseDescriptorDeleter`, `CuSparseDescriptor`, `ConstCuSparseDescriptorDeleter`, `ConstCuSparseDescriptor`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`, `TORCH_CUDA_CPP_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `ATen/cuda/CUDAContext.h`
- `ATen/cuda/CUDASparse.h`
- `c10/core/ScalarType.h`
- `type_traits`


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

Files in the same folder (`aten/src/ATen/cuda`):

- [`CublasHandlePool.cpp_docs.md`](./CublasHandlePool.cpp_docs.md)
- [`llvm_basic.cpp_docs.md`](./llvm_basic.cpp_docs.md)
- [`CUDABlas.h_docs.md`](./CUDABlas.h_docs.md)
- [`jiterator.cu_docs.md`](./jiterator.cu_docs.md)
- [`CUDAGraph.h_docs.md`](./CUDAGraph.h_docs.md)
- [`llvm_jit_strings.h_docs.md`](./llvm_jit_strings.h_docs.md)
- [`llvm_complex.cpp_docs.md`](./llvm_complex.cpp_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md`](./CUDAGeneratorImpl.cpp_docs.md)
- [`cub_definitions.cuh_docs.md`](./cub_definitions.cuh_docs.md)
- [`jiterator_impl.h_docs.md`](./jiterator_impl.h_docs.md)


## Cross-References

- **File Documentation**: `CUDASparseDescriptors.h_docs.md`
- **Keyword Index**: `CUDASparseDescriptors.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/aten/src/ATen/cuda`):

- [`PhiloxCudaState.h_docs.md_docs.md`](./PhiloxCudaState.h_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md_docs.md`](./CUDAGeneratorImpl.cpp_docs.md_docs.md)
- [`Exceptions.cpp_docs.md_docs.md`](./Exceptions.cpp_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_kw.md_docs.md`](./CUDAGeneratorImpl.cpp_kw.md_docs.md)
- [`Sleep.h_docs.md_docs.md`](./Sleep.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-2.cu_kw.md_docs.md`](./cub-RadixSortPairs-int64-2.cu_kw.md_docs.md)
- [`CUDASparseDescriptors.h_kw.md_docs.md`](./CUDASparseDescriptors.h_kw.md_docs.md)
- [`jiterator_impl.h_docs.md_docs.md`](./jiterator_impl.h_docs.md_docs.md)
- [`CUDAContext.h_docs.md_docs.md`](./CUDAContext.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-4.cu_docs.md_docs.md`](./cub-RadixSortPairs-int64-4.cu_docs.md_docs.md)


## Cross-References

- **File Documentation**: `CUDASparseDescriptors.h_docs.md_docs.md`
- **Keyword Index**: `CUDASparseDescriptors.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
