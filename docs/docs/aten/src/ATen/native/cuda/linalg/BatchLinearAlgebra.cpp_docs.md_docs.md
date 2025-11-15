# Documentation: `docs/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp_docs.md`
- **Size**: 53,733 bytes (52.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp`
- **Size**: 110,357 bytes (107.77 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <utility>

#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include <c10/util/Exception.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>
#include <ATen/native/cuda/linalg/MagmaUtils.h>
#include <ATen/native/cpu/zmath.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_solve_triangular.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_linalg_check_errors.h>
#endif

#if AT_MAGMA_ENABLED()
#include <magma_types.h>
#include <magma_v2.h>
#include <ATen/cuda/detail/CUDAHooks.h>

const bool use_magma_ = true;

namespace {
struct MagmaInitializer {
  MagmaInitializer() {
#if defined(BUILD_LAZY_CUDA_LINALG)
    magma_init();
#else
    ::at::cuda::detail::set_magma_init_fn([]{ magma_init(); });
#endif
  }
} initializer;
}  // namespace (anonymous)

#define AT_MAGMA_VERSION MAGMA_VERSION_MAJOR*100 + MAGMA_VERSION_MINOR*10 + MAGMA_VERSION_MICRO

// Check that MAGMA never releases MAGMA_VERSION_MINOR >= 10 or MAGMA_VERSION_MICRO >= 10
#if MAGMA_VERSION_MINOR >= 10 || MAGMA_VERSION_MICRO >= 10
#error "MAGMA release minor or micro version >= 10, please correct AT_MAGMA_VERSION"
#endif

#else
const bool use_magma_ = false;

#endif

namespace at::native {
#if defined(BUILD_LAZY_CUDA_LINALG)
// All registrations with PyTorch runtime should be done dynamically
// so if library is lazy loaded it must not export anything, otherwise
// it can result in symbol clashes
namespace lazy_linalg {
#endif

#if AT_MAGMA_ENABLED()

template <class scalar_t>
void magmaLdlHermitian(
    magma_uplo_t uplo,
    magma_int_t n,
    scalar_t* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  TORCH_CHECK(
      false,
      "LDL decomposition is not available.",
      "Please rebuild with MAGMA 2.5.4+.");
}

template<class scalar_t>
void magmaLu(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info);

template<class scalar_t>
void magmaLuBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaLuNoPiv(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    magma_int_t* info);

template<class scalar_t>
void magmaLuNoPivBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaCholeskySolve(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t* dA, magma_int_t ldda,
    scalar_t* dB, magma_int_t lddb, magma_int_t* info);

template<class scalar_t>
void magmaCholeskySolveBatched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaCholesky(
    magma_uplo_t uplo, magma_int_t n, scalar_t* dA,
    magma_int_t ldda, magma_int_t* info);

template<class scalar_t>
void magmaCholeskyBatched(
    magma_uplo_t uplo, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaTriangularSolveBatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    scalar_t** dA_array, magma_int_t ldda, scalar_t** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);

template<class scalar_t>
inline magma_int_t magmaGeqrfOptimalBlocksize(magma_int_t m, magma_int_t n);

template<class scalar_t>
void magmaGeqrf(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    scalar_t* tau, scalar_t* dT, magma_int_t* info, bool is_v2);

template<class scalar_t, class value_t=scalar_t>
void magmaSyevd(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    value_t* w, scalar_t* wA, magma_int_t ldwa, scalar_t* work, magma_int_t lwork, value_t* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info);

template<class scalar_t, class value_t=scalar_t>
void magmaEig(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, scalar_t *A, magma_int_t lda,
    scalar_t *w, scalar_t *VL, magma_int_t ldvl,
    scalar_t *VR, magma_int_t ldvr, scalar_t *work, magma_int_t lwork,
    value_t *rwork,
    magma_int_t *info);

template<class scalar_t, class value_t=scalar_t>
void magmaSvd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, scalar_t* A,
    magma_int_t lda, value_t* s, scalar_t* U, magma_int_t ldu,
    scalar_t* Vh, magma_int_t ldvh, scalar_t* work, magma_int_t lwork,
    value_t* rwork,
    magma_int_t* iwork, magma_int_t* info);

template<class scalar_t>
void magmaLuSolve(
    magma_int_t n, magma_int_t nrhs, scalar_t* dA, magma_int_t ldda, magma_int_t* ipiv,
    scalar_t* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans);

template<class scalar_t>
void magmaLuSolveBatched(
    magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans);

template<class scalar_t>
void magmaGels(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    scalar_t* dA, magma_int_t ldda, scalar_t* dB, magma_int_t lddb,
    scalar_t* hwork, magma_int_t lwork, magma_int_t* info);

#if AT_MAGMA_VERSION >= 254

template <>
void magmaLdlHermitian<double>(
    magma_uplo_t uplo,
    magma_int_t n,
    double* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dsytrf_gpu(uplo, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaLdlHermitian<float>(
    magma_uplo_t uplo,
    magma_int_t n,
    float* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_ssytrf_gpu(uplo, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaLdlHermitian<c10::complex<double>>(
    magma_uplo_t uplo,
    magma_int_t n,
    c10::complex<double>* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zhetrf_gpu(
      uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaLdlHermitian<c10::complex<float>>(
    magma_uplo_t uplo,
    magma_int_t n,
    c10::complex<float>* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_chetrf_gpu(
      uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

#endif // AT_MAGMA_VERSION >= 254

template<>
void magmaLu<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgetrf_gpu(m, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgetrf_gpu(m, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgetrf_gpu(m, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgetrf_gpu(m, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_dgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_sgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_zgetrf_batched(m, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_cgetrf_batched(m, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgetrf_nopiv_gpu(m, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgetrf_nopiv_gpu(m, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgetrf_nopiv_gpu(m, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgetrf_nopiv_gpu(m, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_dgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_sgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_zgetrf_nopiv_batched(m, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_cgetrf_nopiv_batched(m, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda,
    double* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dpotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda,
    float* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_spotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda,
    c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zpotrs_gpu(uplo, n, nrhs,
    reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
    reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda,
    c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cpotrs_gpu(uplo, n, nrhs,
    reinterpret_cast<magmaFloatComplex*>(dA), ldda,
    reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    double** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_dpotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    float** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_spotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<double>** dA_array, magma_int_t ldda,
    c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_zpotrs_batched(uplo, n, nrhs,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_cpotrs_batched(uplo, n, nrhs,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<double>(
    magma_uplo_t uplo, magma_int_t n, double* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dpotrf_gpu(uplo, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<float>(
    magma_uplo_t uplo, magma_int_t n, float* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_spotrf_gpu(uplo, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<double>* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zpotrf_gpu(uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<float>* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cpotrf_gpu(uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<double>(
    magma_uplo_t uplo, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_dpotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<float>(
    magma_uplo_t uplo, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_spotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_zpotrf_batched(uplo, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_cpotrf_batched(uplo, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<double>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    double** dA_array, magma_int_t ldda, double** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmablas_dtrsm_batched(side, uplo, trans, diag, m, n, 1, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<float>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t ldda, float** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmablas_strsm_batched(side, uplo, trans, diag, m, n, 1, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<c10::complex<double>>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<double>** dA_array, magma_int_t ldda, c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmaDoubleComplex alpha({1, 0});
  magmablas_ztrsm_batched(side, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<c10::complex<float>>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<float>** dA_array, magma_int_t ldda, c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmaFloatComplex alpha({1, 0});
  magmablas_ctrsm_batched(side, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
inline magma_int_t magmaGeqrfOptimalBlocksize<double>(magma_int_t m, magma_int_t n) {
  return magma_get_dgeqrf_nb(m, n);
}

template<>
inline magma_int_t magmaGeqrfOptimalBlocksize<float>(magma_int_t m, magma_int_t n) {
  return magma_get_sgeqrf_nb(m, n);
}

template <>
inline magma_int_t magmaGeqrfOptimalBlocksize<c10::complex<double>>(
    magma_int_t m,
    magma_int_t n) {
  return magma_get_zgeqrf_nb(m, n);
}

template <>
inline magma_int_t magmaGeqrfOptimalBlocksize<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n) {
  return magma_get_cgeqrf_nb(m, n);
}

template<>
void magmaGeqrf<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    double* tau, double* dT, magma_int_t* info, bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_dgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  } else {
    magma_dgeqrf2_gpu(m, n, dA, ldda, tau, info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGeqrf<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    float* tau, float* dT, magma_int_t* info, bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_sgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  } else {
    magma_sgeqrf2_gpu(m, n, dA, ldda, tau, info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGeqrf<c10::complex<double>>(
    magma_int_t m,
    magma_int_t n,
    c10::complex<double>* dA,
    magma_int_t ldda,
    c10::complex<double>* tau,
    c10::complex<double>* dT,
    magma_int_t* info,
    bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_zgeqrf_gpu(
        m,
        n,
        reinterpret_cast<magmaDoubleComplex*>(dA),
        ldda,
        reinterpret_cast<magmaDoubleComplex*>(tau),
        reinterpret_cast<magmaDoubleComplex*>(dT),
        info);
  } else {
    magma_zgeqrf2_gpu(
        m,
        n,
        reinterpret_cast<magmaDoubleComplex*>(dA),
        ldda,
        reinterpret_cast<magmaDoubleComplex*>(tau),
        info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGeqrf<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n,
    c10::complex<float>* dA,
    magma_int_t ldda,
    c10::complex<float>* tau,
    c10::complex<float>* dT,
    magma_int_t* info,
    bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_cgeqrf_gpu(
        m,
        n,
        reinterpret_cast<magmaFloatComplex*>(dA),
        ldda,
        reinterpret_cast<magmaFloatComplex*>(tau),
        reinterpret_cast<magmaFloatComplex*>(dT),
        info);
  } else {
    magma_cgeqrf2_gpu(
        m,
        n,
        reinterpret_cast<magmaFloatComplex*>(dA),
        ldda,
        reinterpret_cast<magmaFloatComplex*>(tau),
        info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<double>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, double* dA, magma_int_t ldda,
    double* w, double* wA, magma_int_t ldwa, double* work, magma_int_t lwork, double* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  MagmaStreamSyncGuard guard;
  magma_dsyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<float>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, float* dA, magma_int_t ldda,
    float* w, float* wA, magma_int_t ldwa, float* work, magma_int_t lwork, float* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  MagmaStreamSyncGuard guard;
  magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<c10::complex<double>, double>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    double* w, c10::complex<double>* wA, magma_int_t ldwa, c10::complex<double>* work, magma_int_t lwork, double* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zheevd_gpu(
      jobz, uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, w, reinterpret_cast<magmaDoubleComplex*>(wA),
      ldwa, reinterpret_cast<magmaDoubleComplex*>(work), lwork, rwork, lrwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<c10::complex<float>, float>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    float* w, c10::complex<float>* wA, magma_int_t ldwa, c10::complex<float>* work, magma_int_t lwork, float* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cheevd_gpu(
      jobz, uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, w, reinterpret_cast<magmaFloatComplex*>(wA),
      ldwa, reinterpret_cast<magmaFloatComplex*>(work), lwork, rwork, lrwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaEig<double>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    double *A, magma_int_t lda,
    double *w,
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    double *work, magma_int_t lwork,
    double *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  // magma [sd]geev wants to separate output arrays: wr and wi for the real
  // and imaginary parts
  double *wr = w;
  double *wi = w + n;
  (void)rwork; // unused
  magma_dgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaEig<float>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    float *A, magma_int_t lda,
    float *w,
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  float *wr = w;
  float *wi = w + n;
  (void)rwork; // unused
  magma_sgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaEig<c10::complex<double>, double>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    c10::complex<double> *A, magma_int_t lda,
    c10::complex<double> *w,
    c10::complex<double> *VL, magma_int_t ldvl,
    c10::complex<double> *VR, magma_int_t ldvr,
    c10::complex<double> *work, magma_int_t lwork,
    double *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  magma_zgeev(jobvl, jobvr, n,
         reinterpret_cast<magmaDoubleComplex*>(A), lda,
         reinterpret_cast<magmaDoubleComplex*>(w),
         reinterpret_cast<magmaDoubleComplex*>(VL), ldvl,
         reinterpret_cast<magmaDoubleComplex*>(VR), ldvr,
         reinterpret_cast<magmaDoubleComplex*>(work), lwork,
         rwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaEig<c10::complex<float>, float>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    c10::complex<float> *A, magma_int_t lda,
    c10::complex<float> *w,
    c10::complex<float> *VL, magma_int_t ldvl,
    c10::complex<float> *VR, magma_int_t ldvr,
    c10::complex<float> *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  magma_cgeev(jobvl, jobvr, n,
         reinterpret_cast<magmaFloatComplex*>(A), lda,
         reinterpret_cast<magmaFloatComplex*>(w),
         reinterpret_cast<magmaFloatComplex*>(VL), ldvl,
         reinterpret_cast<magmaFloatComplex*>(VR), ldvr,
         reinterpret_cast<magmaFloatComplex*>(work), lwork,
         rwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<double>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, double* A,
    magma_int_t lda, double* s, double* U, magma_int_t ldu,
    double* Vh, magma_int_t ldvh, double* work, magma_int_t lwork,
    double *rwork, magma_int_t* iwork, magma_int_t* info) {
  (void)rwork; // unused
  MagmaStreamSyncGuard guard;
  magma_dgesdd(jobz, m, n, A, lda, s, U, ldu, Vh, ldvh, work, lwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<float>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, float* A,
    magma_int_t lda, float* s, float* U, magma_int_t ldu,
    float* Vh, magma_int_t ldvh, float* work, magma_int_t lwork,
    float* rwork, magma_int_t* iwork, magma_int_t* info) {
  (void)rwork; // unused
  MagmaStreamSyncGuard guard;
  magma_sgesdd(jobz, m, n, A, lda, s, U, ldu, Vh, ldvh, work, lwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<c10::complex<float>, float>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, c10::complex<float>* A,
    magma_int_t lda, float* s, c10::complex<float>* U, magma_int_t ldu,
    c10::complex<float>* Vh, magma_int_t ldvh, c10::complex<float>* work, magma_int_t lwork,
    float *rwork, magma_int_t* iwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgesdd(jobz, m, n, reinterpret_cast<magmaFloatComplex*>(A), lda, s,
                reinterpret_cast<magmaFloatComplex*>(U), ldu,
                reinterpret_cast<magmaFloatComplex*>(Vh), ldvh,
                reinterpret_cast<magmaFloatComplex*>(work), lwork,
                rwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<c10::complex<double>, double>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, c10::complex<double>* A,
    magma_int_t lda, double* s, c10::complex<double>* U, magma_int_t ldu,
    c10::complex<double>* Vh, magma_int_t ldvh, c10::complex<double>* work, magma_int_t lwork,
    double *rwork, magma_int_t* iwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgesdd(jobz, m, n, reinterpret_cast<magmaDoubleComplex*>(A), lda, s,
                reinterpret_cast<magmaDoubleComplex*>(U), ldu,
                reinterpret_cast<magmaDoubleComplex*>(Vh), ldvh,
                reinterpret_cast<magmaDoubleComplex*>(work), lwork,
                rwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<double>(
    magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda, magma_int_t* ipiv,
    double* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  MagmaStreamSyncGuard guard;
  magma_dgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<float>(
    magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda, magma_int_t* ipiv,
    float* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  MagmaStreamSyncGuard guard;
  magma_sgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda, magma_int_t* ipiv,
    c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  MagmaStreamSyncGuard guard;
  magma_zgetrs_gpu(trans, n, nrhs, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda, magma_int_t* ipiv,
    c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  MagmaStreamSyncGuard guard;
  magma_cgetrs_gpu(trans, n, nrhs, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    double** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
  info = magma_dgetrs_batched(trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    float** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
 info = magma_sgetrs_batched(trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchsize, magma_queue.get_queue());
 AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
  info = magma_zgetrs_batched(trans, n, nrhs, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
 info = magma_cgetrs_batched(trans, n, nrhs, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
 AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGels<float>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    float* dA, magma_int_t ldda, float* dB, magma_int_t lddb,
    float* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgels_gpu(trans, m, n, nrhs,
      dA, ldda, dB, lddb,
      hwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGels<double>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    double* dA, magma_int_t ldda, double* dB, magma_int_t lddb,
    double* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgels_gpu(trans, m, n, nrhs,
      dA, ldda, dB, lddb,
      hwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGels<c10::complex<float>>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    c10::complex<float>* dA, magma_int_t ldda, c10::complex<float>* dB, magma_int_t lddb,
    c10::complex<float>* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgels_gpu(trans, m, n, nrhs,
      reinterpret_cast<magmaFloatComplex*>(dA), ldda,
      reinterpret_cast<magmaFloatComplex*>(dB), lddb,
      reinterpret_cast<magmaFloatComplex*>(hwork), lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGels<c10::complex<double>>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    c10::complex<double>* dA, magma_int_t ldda, c10::complex<double>* dB, magma_int_t lddb,
    c10::complex<double>* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgels_gpu(trans, m, n, nrhs,
      reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
      reinterpret_cast<magmaDoubleComplex*>(dB), lddb,
      reinterpret_cast<magmaDoubleComplex*>(hwork), lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

namespace {

/*
  MAGMA can return errors both as a return value and in the info argument.
  The return value and info should always be identical.
  In general, the meaning is as given in this table.
  Predefined error codes are large negative numbers. Using the symbolic
  constants below is preferred, but the numeric values can be found in
  include/magma_types.h.

  Info                       |  Description
  -----------                |  -----------
  info = 0 (MAGMA_SUCCESS)   |  Successful exit
  info < 0, but small        |  For info = -i, the i-th argument had an illegal value
  info > 0                   |  Function-specific error such as singular matrix
  MAGMA_ERR_DEVICE_ALLOC     |  Could not allocate GPU device memory
  MAGMA_ERR_HOST_ALLOC       |  Could not allocate CPU host memory
  MAGMA_ERR_ILLEGAL_VALUE    |  An argument had an illegal value (deprecated; instead it should return -i to say the i-th argument was bad)
  MAGMA_ERR_INVALID_PTR      |  Can't free pointer
  MAGMA_ERR_NOT_IMPLEMENTED  |  Function or option not implemented
  MAGMA_ERR_NOT_SUPPORTED    |  Function or option not supported on the current architecture
*/
void checkMagmaInternalError(magma_int_t info, const std::string& magma_function_name) {
  // if info > 0 the error is function-specific, do nothing in this case
  TORCH_CHECK(info >= 0,
      "MAGMA error: ",
      magma_strerror(info),
      ", info = ", info,
      ", when calling ", magma_function_name);
}

magma_trans_t to_magma(TransposeType trans) {
  switch (trans) {
    case TransposeType::NoTranspose: return MagmaNoTrans;
    case TransposeType::Transpose: return MagmaTrans;
    case TransposeType::ConjTranspose: return MagmaConjTrans;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
} // anonymous namespace
#endif // AT_MAGMA_ENABLED()

#define ALLOCATE_ARRAY(name, type, size) \
  auto storage_##name = pin_memory<type>(size); \
  name = static_cast<type*>(storage_##name.mutable_data());

namespace {

template <typename scalar_t>
void apply_ldl_factor_magma(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& info,
    bool upper) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "torch.linalg.ldl_factor: MAGMA library not found in "
      "compilation. Please rebuild with MAGMA.");
#else
  auto batch_size = batchCount(A);
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t leading_dim = magma_int_cast(A.stride(-1), "A.stride(-1)");
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  auto a_data = A.mutable_data_ptr<scalar_t>();
  Tensor pivots_cpu =
      at::empty_like(pivots, pivots.options().device(kCPU).pinned_memory(true));
  auto pivots_data = pivots_cpu.mutable_data_ptr<magma_int_t>();
  Tensor info_cpu =
      at::empty_like(info, info.options().device(kCPU).pinned_memory(true));
  auto info_data = info_cpu.mutable_data_ptr<magma_int_t>();

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* a_working_ptr = &a_data[i * a_stride];
    magma_int_t* pivots_working_ptr = &pivots_data[i * pivots_stride];
    magma_int_t* info_working_ptr = &info_data[i];
    magmaLdlHermitian<scalar_t>(
        uplo,
        n,
        a_working_ptr,
        leading_dim,
        pivots_working_ptr,
        info_working_ptr);
  }
  pivots.copy_(pivots_cpu);
  info.copy_(info_cpu);
#endif
}

void ldl_factor_magma(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  if (LD.is_complex()) {
    TORCH_CHECK(
        hermitian,
        "torch.linalg.ldl_factor: complex tensors with hermitian=False flag are not supported with MAGMA backend. ",
        "Currently preferred backend is ",
        at::globalContext().linalgPreferredBackend(),
        ", please set 'default' or 'cusolver' backend with torch.backends.cuda.preferred_linalg_library");
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_factor_magma", [&] {
        apply_ldl_factor_magma<scalar_t>(LD, pivots, info, upper);
      });
}

void ldl_factor_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
       { ldl_factor_cusolver(
          LD, pivots, info, upper, hermitian);
        return;
}
    case at::LinalgBackend::Magma:
       { ldl_factor_magma(LD, pivots, info, upper, hermitian);
        return;
}
    default:
    // By default use cusolver if available and magma otherwise.
    // If cusolver and magma 2.5.4+ are both available and hermitian=true,
    // call magma for complex inputs
#ifdef USE_LINALG_SOLVER
#if AT_MAGMA_ENABLED() && (AT_MAGMA_VERSION >= 254)
      if (LD.is_complex() && hermitian) {
        return ldl_factor_magma(
            LD, pivots, info, upper, hermitian);
      }
#endif
    { ldl_factor_cusolver(
      LD, pivots, info, upper, hermitian);
      return;
    }
#else
      return ldl_factor_magma(LD, pivots, info, upper, hermitian);
#endif
  }
}

void ldl_solve_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  // TODO: It should be possible to add the MAGMA backend for this function when using MAGMA 2.6.0
  // https://bitbucket.org/icl/magma/src/c703d112dcf19eb8c73676cef10888aa2ef73457/ReleaseNotes#lines-48
  if (LD.is_complex()) {
    TORCH_CHECK(
        !hermitian,
        "torch.linalg.ldl_solve: complex tensors with hermitian=True flag are not supported on CUDA.");
  }

  ldl_solve_cusolver(LD, pivots, B, upper);
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_CUDA_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, int64_t& info) {
#if !AT_MAGMA_ENABLED()
TORCH_CHECK(false, "cholesky_solve: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t lda = std::max<magma_int_t>(1, n);
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  int info_tmp = 0;
  if (b.dim() == 2) {
    magmaCholeskySolve<scalar_t>(uplo, n, nrhs, A_data, lda,
                                 b_data, lda, &info_tmp);
    info = info_tmp;
  } else {
    auto A_mat_stride = matrixStride(A);
    auto b_mat_stride = matrixStride(b);
    magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");

    scalar_t** A_array;
    scalar_t** b_array;

    ALLOCATE_ARRAY(A_array, scalar_t*, batch_size);
    ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      A_array[i] = &A_data[i * A_mat_stride];
      b_array[i] = &b_data[i * b_mat_stride];
    }

    MAGMAQueue magma_queue(b.get_device());

    constexpr int64_t batch_limit = 65535;
    // Compute as many batches of 65535 possible
    // The number of "mini"-batches are floor(batch_size / batch_limit)
    // and these cover floor(batch_size / batch_limit) * batch_limit matrix solves
    int64_t mini_batches = batch_size / batch_limit, mini_idx;
    for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
      scalar_t** A_array_cur = &A_array[mini_idx];
      scalar_t** b_array_cur = &b_array[mini_idx];

      magmaCholeskySolveBatched<scalar_t>(
          uplo, n, nrhs, A_array_cur, lda, b_array_cur, lda,
          info_tmp, batch_limit, magma_queue);

      if (info_tmp != 0) {
        break;
      }
    }

    // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
    // which concisely is equal to batch_size % batch_limit
    if (batch_size % batch_limit != 0 && info_tmp == 0) {
      magmaCholeskySolveBatched<scalar_t>(
          uplo, n, nrhs, &A_array[mini_idx], lda, &b_array[mini_idx], lda,
          info_tmp, batch_size % batch_limit, magma_queue);
    }

    info = info_tmp;
  }
#endif
}

Tensor _cholesky_solve_helper_cuda_magma(const Tensor& self, const Tensor& A, bool upper) {
  int64_t info = 0;
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_solve_cuda", [&]{
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, info);
  });
  TORCH_CHECK(info == 0, "MAGMA cholesky_solve : invalid argument: ", -info);
  return self_working_copy;
}

// Todo: cusolverDn<T>potrsBatched only supports nrhs == 1 and does not have good performance.
//     Batched cholesky_solve is dispatched to magma.
Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
#if defined(USE_LINALG_SOLVER)
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
      return _cholesky_solve_helper_cuda_cusolver(self, A, upper);
    case at::LinalgBackend::Magma:
      return _cholesky_solve_helper_cuda_magma(self, A, upper);
    default:
      if (batchCount(self) == 1 || !use_magma_) {
        return _cholesky_solve_helper_cuda_cusolver(self, A, upper);
      } else {
        return _cholesky_solve_helper_cuda_magma(self, A, upper);
      }
  }
#else
  return _cholesky_solve_helper_cuda_magma(self, A, upper);
#endif
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_cholesky(const Tensor& self, bool upper, const Tensor& info) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.cholesky on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please use PyTorch built with MAGMA support.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto self_data = self.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");
  auto lda = std::max<magma_int_t>(1, n);

  if (self.dim() == 2) {
    // magmaCholesky requires info to be on CPU
    magma_int_t info_cpu = 0;
    magmaCholesky<scalar_t>(uplo, n, self_data, lda, &info_cpu);
    info.fill_(info_cpu);
  } else {
    TORCH_INTERNAL_ASSERT(info.is_cuda());
    auto info_data = info.data_ptr<magma_int_t>();

    // magmaCholeskyBatched supports only upper=false
    uplo = MagmaLower;

    auto self_mat_stride = matrixStride(self);
    magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");

    scalar_t** self_array;

    ALLOCATE_ARRAY(self_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      self_array[i] = &self_data[i * self_mat_stride];
    }

    MAGMAQueue magma_queue(self.get_device());

    // Compute as many batches of 262140 possible
    // 262140 is the size of the largest batch of matrices that can be run with
    // violating maximum kernel configuration
    // For complex input the batch limit is 65535 (determined experimentally, see https://github.com/pytorch/pytorch/pull/47047#discussion_r516086923 for more information)
    int64_t batch_limit = self.is_complex() ? 65535 : 262140;

    for (int64_t mini_idx = 0; mini_idx < batch_size; mini_idx += batch_limit) {
      int64_t nbatches = std::min(batch_limit, batch_size - mini_idx);
      scalar_t** self_array_cur = &self_array[mini_idx];
      magma_int_t* info_array_cur = &info_data[mini_idx];

      magmaCholeskyBatched<scalar_t>(
        uplo, n, self_array_cur, lda, info_array_cur, nbatches, magma_queue);
    }
  }
#endif
}

void cholesky_helper_magma(const Tensor& input, bool upper, const Tensor& info) {
  Tensor result = input;
  if (input.dim() > 2) {
    // MAGMA's batched cholesky operator has an off-by-one error causing IMA
    // (see https://github.com/pytorch/pytorch/issues/42666). This code is based
    // on the #cloneBatchedColumnMajor function however it pads the input with
    // one extra element utilizing the fact that the resize_as_ method preserves
    // the storage even if it's lar
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cuda/linalg`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cuda/linalg`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/cuda/linalg`):

- [`CUDASolver.h_kw.md_docs.md`](./CUDASolver.h_kw.md_docs.md)
- [`MagmaUtils.h_kw.md_docs.md`](./MagmaUtils.h_kw.md_docs.md)
- [`CUDASolver.cpp_docs.md_docs.md`](./CUDASolver.cpp_docs.md_docs.md)
- [`CUDASolver.cpp_kw.md_docs.md`](./CUDASolver.cpp_kw.md_docs.md)
- [`CudssHandlePool.cpp_kw.md_docs.md`](./CudssHandlePool.cpp_kw.md_docs.md)
- [`BatchLinearAlgebraLibBlas.cpp_kw.md_docs.md`](./BatchLinearAlgebraLibBlas.cpp_kw.md_docs.md)
- [`BatchLinearAlgebraLib.h_docs.md_docs.md`](./BatchLinearAlgebraLib.h_docs.md_docs.md)
- [`CusolverDnHandlePool.cpp_docs.md_docs.md`](./CusolverDnHandlePool.cpp_docs.md_docs.md)
- [`BatchLinearAlgebraLibBlas.cpp_docs.md_docs.md`](./BatchLinearAlgebraLibBlas.cpp_docs.md_docs.md)
- [`BatchLinearAlgebraLib.h_kw.md_docs.md`](./BatchLinearAlgebraLib.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `BatchLinearAlgebra.cpp_docs.md_docs.md`
- **Keyword Index**: `BatchLinearAlgebra.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
