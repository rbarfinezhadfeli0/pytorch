# Documentation: `docs/aten/src/ATen/native/CPUBlas.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/CPUBlas.h_docs.md`
- **Size**: 11,357 bytes (11.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/CPUBlas.h`

## File Metadata

- **Path**: `aten/src/ATen/native/CPUBlas.h`
- **Size**: 8,864 bytes (8.66 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TransposeType.h>
#include <c10/util/complex.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Scalar.h>


namespace at::native::cpublas {

namespace internal {
void normalize_last_dims(
  TransposeType transa, TransposeType transb,
  int64_t m, int64_t n, int64_t k,
  int64_t *lda, int64_t *ldb, int64_t *ldc);
}  // namespace internal

using gemm_fn = void(*)(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc);

DECLARE_DISPATCH(gemm_fn, gemm_stub)

using gemm_no_downcast_fn = void(*)(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc);

DECLARE_DISPATCH(gemm_no_downcast_fn, gemm_no_downcast_stub)

template <typename scalar_t>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<scalar_t> alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    at::opmath_type<scalar_t> beta,
    scalar_t *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  gemm_stub(
    kCPU, c10::CppTypeToScalarType<scalar_t>::value,
    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    const double *a, int64_t lda,
    const double *b, int64_t ldb,
    double beta,
    double *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    float beta,
    at::BFloat16 *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    float beta,
    at::Half *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    c10::complex<double> alpha,
    const c10::complex<double> *a, int64_t lda,
    const c10::complex<double> *b, int64_t ldb,
    c10::complex<double> beta,
    c10::complex<double> *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    c10::complex<float> alpha,
    const c10::complex<float> *a, int64_t lda,
    const c10::complex<float> *b, int64_t ldb,
    c10::complex<float> beta,
    c10::complex<float> *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t alpha,
    const int64_t *a, int64_t lda,
    const int64_t *b, int64_t ldb,
    int64_t beta,
    int64_t *c, int64_t ldc);

template <typename scalar_t>
void gemm_batched(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t * const *a, int64_t lda,
    const scalar_t * const *b, int64_t ldb,
    const scalar_t beta,
    scalar_t * const *c, int64_t ldc);

template <typename scalar_t>
void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c);

using axpy_fn = void(*)(at::ScalarType type, int64_t n, const Scalar& a, const void *x, int64_t incx, void *y, int64_t incy);

DECLARE_DISPATCH(axpy_fn, axpy_stub)

template<typename scalar_t>
void axpy(int64_t n, scalar_t a, const scalar_t *x, int64_t incx, scalar_t *y, int64_t incy){
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  axpy_stub(
      kCPU, c10::CppTypeToScalarType<scalar_t>::value,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, double a, const double *x, int64_t incx, double *y, int64_t incy);
void axpy(int64_t n, float a, const float *x, int64_t incx, float *y, int64_t incy);
void axpy(int64_t n, c10::complex<double> a, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void axpy(int64_t n, c10::complex<float> a, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

using copy_fn = void(*)(at::ScalarType type, int64_t n, const void *x, int64_t incx, void *y, int64_t incy);

DECLARE_DISPATCH(copy_fn, copy_stub)

template<typename scalar_t>
void copy(int64_t n, const scalar_t *x, int64_t incx, scalar_t *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  copy_stub(
      kCPU, c10::CppTypeToScalarType<scalar_t>::value,
      n, x, incx, y, incy);
}

void copy(int64_t n, const double *x, int64_t incx, double *y, int64_t incy);
void copy(int64_t n, const float *x, int64_t incx, float *y, int64_t incy);
void copy(int64_t n, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void copy(int64_t n, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

// Batch-reduce GEMM
// Operates by the following formula:
// C = SUM(A[i] x B[i]) + C if add_C is true, i = 0 to batch size
// A Base pointer to a tensor A.
// B Base pointer to a tensor B.
// C Pointer to a tensor C (accumulation buffer).
// Note only batch size 1 is used currently

// Define macros for available brgemm APIs
// so that callers can determine which APIs are available
#define CPUBLAS_BRGEMM_F16F16F32 // half * half -> float
#define CPUBLAS_BRGEMM_BF16BF16F32 // bfloat16 * bfloat16 -> float
#define CPUBLAS_BRGEMM_F32F32F32 // float * float -> float
#define CPUBLAS_BRGEMM_U8U8I32 // unsigned char * unsigned char -> int32
#define CPUBLAS_BRGEMM_U8I8I32 // unsigned char * signed char -> int32
#define CPUBLAS_BRGEMM_I8I8I32 // signed char * signed char -> int32

TORCH_API void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const at::Half* A,
    const at::Half* B,
    float* C,
    bool is_vnni = true);

TORCH_API void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const at::BFloat16* A,
    const at::BFloat16* B,
    float* C,
    bool is_vnni = true);

TORCH_API void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const float* A,
    const float* B,
    float* C,
    bool is_vnni = false);

TORCH_API void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const unsigned char* A,
    const unsigned char* B,
    int32_t* C,
    bool is_vnni = true);

TORCH_API void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const unsigned char* A,
    const signed char* B,
    int32_t* C,
    bool is_vnni = true);

TORCH_API void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const signed char* A,
    const signed char* B,
    int32_t* C,
    bool is_vnni = true);

// Release brgemm hardware context
TORCH_API void brgemm_release(bool is_vnni = true);

// Pack B matrix to get better performance if needed
TORCH_API void pack(
    int64_t K,
    int64_t N,
    int64_t ld_in,
    int64_t ld_out,
    ScalarType dt_in,
    ScalarType dt_out,
    const void* in,
    void* out);

// Whether pack is supported in the platform.
TORCH_API bool could_pack(ScalarType dt_in);

} // namespace at::native::cpublas

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 33 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `internal`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/OpMathType.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/TransposeType.h`
- `c10/util/complex.h`
- `c10/core/ScalarType.h`
- `c10/core/Scalar.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `CPUBlas.h_docs.md`
- **Keyword Index**: `CPUBlas.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `CPUBlas.h_docs.md_docs.md`
- **Keyword Index**: `CPUBlas.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
