# Documentation: BatchLinearAlgebra.cpp

## File Metadata
- **Path**: `aten/src/ATen/native/BatchLinearAlgebra.cpp`
- **Size**: 169041 bytes
- **Lines**: 4100
- **Extension**: .cpp
- **Type**: Regular file

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>

#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/zmath.h>

#include <c10/util/irange.h>

#include <utility>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_spsolve.h>
#include <ATen/ops/_cholesky_solve_helper.h>
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <ATen/ops/_linalg_eigh.h>
#include <ATen/ops/_linalg_eigh_meta.h>
#include <ATen/ops/_linalg_eigh_native.h>
#include <ATen/ops/_linalg_eigvals.h>
#include <ATen/ops/_linalg_eigvals_native.h>
#include <ATen/ops/_linalg_solve_ex.h>
#include <ATen/ops/_linalg_solve_ex_meta.h>
#include <ATen/ops/_linalg_solve_ex_native.h>
#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_linalg_svd_meta.h>
#include <ATen/ops/_linalg_svd_native.h>
#include <ATen/ops/_lu_with_info_native.h>
#include <ATen/ops/all.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cholesky.h>
#include <ATen/ops/cholesky_inverse.h>
#include <ATen/ops/cholesky_inverse_native.h>
#include <ATen/ops/cholesky_native.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/cholesky_solve_native.h>
#include <ATen/ops/clone.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/geqrf.h>
#include <ATen/ops/geqrf_native.h>
#include <ATen/ops/inverse_native.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <ATen/ops/linalg_cholesky_ex_meta.h>
#include <ATen/ops/linalg_cholesky_ex_native.h>
#include <ATen/ops/linalg_cholesky_native.h>
#include <ATen/ops/linalg_eig.h>
#include <ATen/ops/linalg_eig_native.h>
#include <ATen/ops/linalg_eigh_native.h>
#include <ATen/ops/linalg_eigvals.h>
#include <ATen/ops/linalg_eigvals_native.h>
#include <ATen/ops/linalg_eigvalsh_native.h>
#include <ATen/ops/linalg_householder_product.h>
#include <ATen/ops/linalg_householder_product_native.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#include <ATen/ops/linalg_inv_native.h>
#include <ATen/ops/linalg_ldl_factor_ex.h>
#include <ATen/ops/linalg_ldl_factor_ex_meta.h>
#include <ATen/ops/linalg_ldl_factor_ex_native.h>
#include <ATen/ops/linalg_ldl_factor_native.h>
#include <ATen/ops/linalg_ldl_solve_meta.h>
#include <ATen/ops/linalg_ldl_solve_native.h>
#include <ATen/ops/linalg_lstsq.h>
#include <ATen/ops/linalg_lstsq_native.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_lu_factor_ex_meta.h>
#include <ATen/ops/linalg_lu_factor_ex_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_lu_meta.h>
#include <ATen/ops/linalg_lu_native.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_lu_solve_meta.h>
#include <ATen/ops/linalg_lu_solve_native.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_qr_meta.h>
#include <ATen/ops/linalg_qr_native.h>
#include <ATen/ops/linalg_solve_ex.h>
#include <ATen/ops/linalg_solve_ex_native.h>
#include <ATen/ops/linalg_solve_native.h>
#include <ATen/ops/linalg_solve_triangular_native.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/linalg_svd_native.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_svdvals_native.h>
#include <ATen/ops/linalg_vander_native.h>
#include <ATen/ops/linalg_vecdot_native.h>
#include <ATen/ops/lu_solve_native.h>
#include <ATen/ops/lu_unpack.h>
#include <ATen/ops/lu_unpack_meta.h>
#include <ATen/ops/lu_unpack_native.h>
#include <ATen/ops/orgqr_native.h>
#include <ATen/ops/ormqr_native.h>
#include <ATen/ops/qr_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/svd_native.h>
#include <ATen/ops/triangular_solve_meta.h>
#include <ATen/ops/triangular_solve_native.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/vdot.h>
#include <ATen/ops/zeros.h>
#endif

// First the required LAPACK implementations are registered here.
// A comment above the registered LAPACK routine suggest which batched
// linear algebra function uses that routine
#if AT_BUILD_WITH_LAPACK()

#ifndef _ARMPL_H  // ArmPL's `cblas.h` pulls in these prototypes.
// getrf
extern "C" void zgetrf_(int *m, int *n, std::complex<double> *a, int *lda, int *ipiv, int *info);
extern "C" void cgetrf_(int *m, int *n, std::complex<float> *a, int *lda, int *ipiv, int *info);
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
#endif

// potrs
#if defined(_WIN32) && defined(_M_ARM64)

// The functions zpotrs, cpotrs, dpotrs, and spotrs are not directly available in LAPACKE on Windows on ARM,
// so we need to have wrapper functions to call them.
// The issue on ARM platform can be found below:
// https://community.arm.com/support-forums/f/high-performance-computing-forum/56512/unable-to-use-lapack---potrs-functions

#define LAPACK_COL_MAJOR 102
#define LAPACK_ROW_MAJOR 101

extern "C" int LAPACKE_zpotrs(int matrix_layout, char uplo, int n, int nrhs, const std::complex<double> *a, int lda, std::complex<double> *b, int ldb);
extern "C" int LAPACKE_cpotrs(int matrix_layout, char uplo, int n, int nrhs, const std::complex<float> *a, int lda, std::complex<float> *b, int ldb);
extern "C" int LAPACKE_dpotrs(int matrix_layout, char uplo, int n, int nrhs, const double *a, int lda, double *b, int ldb);
extern "C" int LAPACKE_spotrs(int matrix_layout, char uplo, int n, int nrhs, const float *a, int lda, float *b, int ldb);

static inline void zpotrs_(char *uplo, int *n, int *nrhs, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, int *info) {
  *info = LAPACKE_zpotrs(LAPACK_COL_MAJOR, *uplo, *n, *nrhs, a, *lda, b, *ldb);
}

static inline void cpotrs_(char *uplo, int *n, int *nrhs, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb, int *info) {
  *info = LAPACKE_cpotrs(LAPACK_COL_MAJOR, *uplo, *n, *nrhs, a, *lda, b, *ldb);
}

static inline void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info){
  *info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, *uplo, *n, *nrhs, a, *lda, b, *ldb);
}

static inline void spotrs_(char *uplo, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info) {
  *info = LAPACKE_spotrs(LAPACK_COL_MAJOR, *uplo, *n, *nrhs, a, *lda, b, *ldb);
}

#else

#ifndef _ARMPL_H  // ArmPL's `cblas.h` pulls in these prototypes.
extern "C" void zpotrs_(char *uplo, int *n, int *nrhs, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, int *info);
extern "C" void cpotrs_(char *uplo, int *n, int *nrhs, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb, int *info);
extern "C" void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern "C" void spotrs_(char *uplo, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);
#endif

#endif

#ifndef _ARMPL_H  // ArmPL's `cblas.h` pulls in these prototypes.
// potrf
extern "C" void zpotrf_(char *uplo, int *n, std::complex<double> *a, int *lda, int *info);
extern "C" void cpotrf_(char *uplo, int *n, std::complex<float> *a, int *lda, int *info);
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);

// potri
extern "C" void zpotri_(char *uplo, int *n, std::complex<double> *a, int *lda, int *info);
extern "C" void cpotri_(char *uplo, int *n, std::complex<float> *a, int *lda, int *info);
extern "C" void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotri_(char *uplo, int *n, float *a, int *lda, int *info);

// sytrf
extern "C" void dsytrf_(
    char* uplo,
    int* n,
    double* a,
    int* lda,
    int* ipiv,
    double* work,
    int* lwork,
    int* info);
extern "C" void ssytrf_(
    char* uplo,
    int* n,
    float* a,
    int* lda,
    int* ipiv,
    float* work,
    int* lwork,
    int* info);
extern "C" void zsytrf_(
    char* uplo,
    int* n,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* work,
    int* lwork,
    int* info);
extern "C" void csytrf_(
    char* uplo,
    int* n,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* work,
    int* lwork,
    int* info);

// hetrf
extern "C" void zhetrf_(
    char* uplo,
    int* n,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* work,
    int* lwork,
    int* info);
extern "C" void chetrf_(
    char* uplo,
    int* n,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* work,
    int* lwork,
    int* info);

// sytrs
extern "C" void dsytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    double* a,
    int* lda,
    int* ipiv,
    double* b,
    int* ldb,
    int* info);
extern "C" void ssytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    float* a,
    int* lda,
    int* ipiv,
    float* b,
    int* ldb,
    int* info);
extern "C" void zsytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* b,
    int* ldb,
    int* info);
extern "C" void csytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* b,
    int* ldb,
    int* info);

// hetrs
extern "C" void zhetrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* b,
    int* ldb,
    int* info);
extern "C" void chetrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* b,
    int* ldb,
    int* info);

// geqrf
extern "C" void zgeqrf_(int *m, int *n, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);
extern "C" void cgeqrf_(int *m, int *n, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
extern "C" void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern "C" void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

// orgqr
extern "C" void zungqr_(int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);
extern "C" void cungqr_(int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
extern "C" void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern "C" void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
#endif

// ormqr
#if defined(_WIN32) && defined(_M_ARM64)

// The functions zunmqr, cunmqr, dormqr, and sormqr are not directly available in LAPACKE on Windows on ARM,
// so we need to have wrapper functions to call them.
// The issue on ARM platform can be found below:
// https://community.arm.com/support-forums/f/high-performance-computing-forum/56512/unable-to-use-lapack---potrs-functions

extern "C" int LAPACKE_zunmqr_work(int matrix_layout, char side, char trans, int m, int n, int k, const std::complex<double> *a, int lda, const std::complex<double> *tau, std::complex<double> *c, int ldc, std::complex<double> *work, int lwork);
extern "C" int LAPACKE_cunmqr_work(int matrix_layout, char side, char trans, int m, int n, int k, const std::complex<float> *a, int lda, const std::complex<float> *tau, std::complex<float> *c, int ldc, std::complex<float> *work, int lwork);
extern "C" int LAPACKE_dormqr_work(int matrix_layout, char side, char trans, int m, int n, int k, const double *a, int lda, const double *tau, double *c, int ldc, double *work, int lwork);
extern "C" int LAPACKE_sormqr_work(int matrix_layout, char side, char trans, int m, int n, int k, const float *a, int lda, const float *tau, float *c, int ldc, float *work, int lwork);

static inline void zunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *c, int *ldc, std::complex<double> *work, int *lwork, int *info) {
    *info = LAPACKE_zunmqr_work(LAPACK_COL_MAJOR, *side, *trans, *m, *n, *k, a, *lda, tau, c, *ldc, work, *lwork);
}

static inline void cunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *c, int *ldc, std::complex<float> *work, int *lwork, int *info) {
    *info = LAPACKE_cunmqr_work(LAPACK_COL_MAJOR, *side, *trans, *m, *n, *k, a, *lda, tau, c, *ldc, work, *lwork);
}

static inline void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info) {
    *info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, *side, *trans, *m, *n, *k, a, *lda, tau, c, *ldc, work, *lwork);
}

static inline void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *lda, float *tau, float *c, int *ldc, float *work, int *lwork, int *info) {
    *info = LAPACKE_sormqr_work(LAPACK_COL_MAJOR, *side, *trans, *m, *n, *k, a, *lda, tau, c, *ldc, work, *lwork);
}
#else
#ifndef _ARMPL_H  // ArmPL's `cblas.h` pulls in these prototypes.
extern "C" void zunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *c, int *ldc, std::complex<double> *work, int *lwork, int *info);
extern "C" void cunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *c, int *ldc, std::complex<float> *work, int *lwork, int *info);
extern "C" void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);
extern "C" void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *lda, float *tau, float *c, int *ldc, float *work, int *lwork, int *info);
#endif
#endif
#ifndef _ARMPL_H  // ArmPL's `cblas.h` pulls in these prototypes.
// syevd
extern "C" void zheevd_(char *jobz, char *uplo, int *n, std::complex<double> *a, int *lda, double *w, std::complex<double> *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);
extern "C" void cheevd_(char *jobz, char *uplo, int *n, std::complex<float> *a, int *lda, float *w, std::complex<float> *work, int *lwork, float *rwork, int *lrwork, int *iwork, int *liwork, int *info);
extern "C" void dsyevd_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *iwork, int *liwork, int *info);
extern "C" void ssyevd_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *iwork, int *liwork, int *info);

// geev
extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double* vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
extern "C" void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr, float *wi, float* vl, int *ldvl, float *vr, int *ldvr, float *work, int *lwork, int *info);
extern "C" void cgeev_(char *jobvl, char *jobvr, int *n,
             std::complex<float> *a, int *lda,
             std::complex<float> *w,
             std::complex<float> *vl, int *ldvl,
             std::complex<float> *vr, int *ldvr,
             std::complex<float> *work, int *lwork,
             float *rwork,
             int *info);
extern "C" void zgeev_(char *jobvl, char *jobvr, int *n,
             std::complex<double> *a, int *lda,
             std::complex<double> *w,
             std::complex<double> *vl, int *ldvl,
             std::complex<double> *vr, int *ldvr,
             std::complex<double> *work, int *lwork,
             double *rwork,
             int *info);

// gesdd
extern "C" void zgesdd_(char *jobz, int *m, int *n, std::complex<double> *a, int *lda,
                        double *s, std::complex<double> *u, int *ldu, std::complex<double> *vt, int *ldvt, std::complex<double> *work, int *lwork, double *rwork, int *iwork, int *info);
extern "C" void cgesdd_(char *jobz, int *m, int *n, std::complex<float> *a, int *lda,
                        float *s, std::complex<float> *u, int *ldu, std::complex<float> *vt, int *ldvt, std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);
extern "C" void dgesdd_(char *jobz, int *m, int *n, double *a, int *lda,
                        double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *iwork, int *info);
extern "C" void sgesdd_(char *jobz, int *m, int *n, float *a, int *lda,
                        float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info);

// getrs
extern "C" void zgetrs_(char *trans, int *n, int *nrhs, std::complex<double> *a, int *lda, int *ipiv, std::complex<double> *b, int *ldb, int *info);
extern "C" void cgetrs_(char *trans, int *n, int *nrhs, std::complex<float> *a, int *lda, int *ipiv, std::complex<float> *b, int *ldb, int *info);
extern "C" void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern "C" void sgetrs_(char *trans, int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);

// gels
extern "C" void zgels_(char *trans, int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    std::complex<double> *work, int *lwork, int *info);
extern "C" void cgels_(char *trans, int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    std::complex<float> *work, int *lwork, int *info);
extern "C" void dgels_(char *trans, int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *work, int *lwork, int *info);
extern "C" void sgels_(char *trans, int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *work, int *lwork, int *info);

// gelsd
extern "C" void zgelsd_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    double *s, double *rcond, int *rank,
    std::complex<double> *work, int *lwork, double *rwork, int *iwork, int *info);
extern "C" void cgelsd_(int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    float *s, float *rcond, int *rank,
    std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);
extern "C" void dgelsd_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *s, double *rcond, int *rank,
    double *work, int *lwork, int *iwork, int *info);
extern "C" void sgelsd_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *s, float *rcond, int *rank,
    float *work, int *lwork, int *iwork, int *info);

// gelsy
extern "C" void zgelsy_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    int *jpvt, double *rcond, int *rank,
    std::complex<double> *work, int *lwork,
    double *rwork, int *info);
extern "C" void cgelsy_(int *m, int *n, int *nrhs,
    std::complex<float> * a, int *lda, std::complex<float> *b, int *ldb,
    int *jpvt, float *rcond, int *rank,
    std::complex<float> *work, int *lwork,
    float *rwork, int *info);
extern "C" void dgelsy_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    int *jpvt, double *rcond, int *rank,
    double *work, int *lwork, int *info);
extern "C" void sgelsy_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    int *jpvt, float *rcond, int *rank,
    float *work, int *lwork, int *info);

// gelss
extern "C" void zgelss_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    double *s, double *rcond, int *rank,
    std::complex<double> *work, int *lwork,
    double *rwork, int *info);
extern "C" void cgelss_(int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    float *s, float *rcond, int *rank,
    std::complex<float> *work, int *lwork,
    float *rwork, int *info);
extern "C" void dgelss_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *s, double *rcond, int *rank,
    double *work, int *lwork, int *info);
extern "C" void sgelss_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *s, float *rcond, int *rank,
    float *work, int *lwork, int *info);
#endif
#endif

#if AT_BUILD_WITH_BLAS()
// trsm
#ifndef _ARMPL_H  // ArmPL's `cblas.h` pulls in these prototypes.
extern "C" void ztrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<double> *alpha, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb);
extern "C" void ctrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<float> *alpha, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb);
extern "C" void dtrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, double *alpha, double *a, int *lda, double *b, int *ldb);
extern "C" void strsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, float *alpha, float *a, int *lda, float *b, int *ldb);
#endif
#endif

namespace at::meta {

TORCH_META_FUNC(linalg_ldl_factor_ex)
(const Tensor& self, bool hermitian, bool check_errors) {
  at::native::squareCheckInputs(self, "torch.linalg.ldl_factor_ex");
  at::native::checkFloatingOrComplex(self, "torch.linalg.ldl_factor_ex");

  auto shape = self.sizes();
  auto ndim = shape.size();

  // prefer column major strides
  auto ld_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig=*/true);
  set_output_strided(0, shape, ld_strides, self.options(), {}); // LD

  set_output_contiguous(
      1, shape.slice(0, ndim - 1), self.options().dtype(ScalarType::Int)); // pivots

  set_output_contiguous(
      2, shape.slice(0, ndim - 2), self.options().dtype(ScalarType::Int)); // info
}

TORCH_META_FUNC(linalg_ldl_solve)
(const Tensor& LD,
 const Tensor& pivots,
 const Tensor& B,
 bool hermitian) {
  at::native::squareCheckInputs(LD, "torch.linalg.ldl_solve");
  at::native::checkFloatingOrComplex(LD, "torch.linalg.ldl_solve");
  at::native::linearSolveCheckInputs(B, LD, "torch.linalg.ldl_solve");
  TORCH_CHECK(
      B.dim() >= 2,
      "torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, but it has ",
      B.dim(),
      " dimensions instead");
  auto expected_pivots_shape = LD.sizes().slice(0, LD.dim() - 1);
  TORCH_CHECK(
      expected_pivots_shape.equals(pivots.sizes()),
      "torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, but got pivots with shape ",
      pivots.sizes(),
      " instead");
  // pivots is allowed to be any integer type
  // LAPACK we use is 32-bit interface while cuSOLVER uses 64-bit interface for integers
  TORCH_CHECK(
      at::isIntegralType(pivots.scalar_type(), /*includeBool=*/false),
      "torch.linalg.ldl_solve: Expected pivots to be integers. Got ",
      pivots.scalar_type());
  TORCH_CHECK(
      LD.scalar_type() == B.scalar_type(),
      "torch.linalg.ldl_solve: ",
      "LD dtype",
      LD.scalar_type(),
      " does not match b dtype ",
      B.scalar_type());

    auto [B_broadcast_size, _] = at::native::_linalg_broadcast_batch_dims(B, LD);

  // prefer column major strides
  auto result_strides = at::native::batched_matrix_contiguous_strides(B_broadcast_size, /*f_contig=*/true);
  set_output_strided(0, B_broadcast_size, result_strides, B.options(), {});
}

TORCH_META_FUNC(triangular_solve)(const Tensor& self, const Tensor& A, bool upper, bool transpose, bool unitriangular) {
  TORCH_CHECK(self.dim() >= 2,
           "torch.triangular_solve: Expected b to have at least 2 dimensions, but it has ", self.dim(), " dimensions instead");
  TORCH_CHECK(A.dim() >= 2,
           "torch.triangular_solve: Expected A to have at least 2 dimensions, but it has ", A.dim(), " dimensions instead");

  at::native::linearSolveCheckInputs(self, A, "triangular_solve");

  if (A.layout() == Layout::Strided) {
    auto [self_broadcast_size, A_broadcast_size] = at::native::_linalg_broadcast_batch_dims(self, A);

    // make column major strides for BLAS
    const auto solution_strides = at::native::batched_matrix_contiguous_strides(self_broadcast_size, /*f-contig=*/true);
    set_output_raw_strided(0, self_broadcast_size, solution_strides, self.options(), {});

    // make column major strides for BLAS
    auto clone_A_strides = at::native::batched_matrix_contiguous_strides(A_broadcast_size, /*f_contig=*/true);
    set_output_raw_strided(1, A_broadcast_size, clone_A_strides, A.options(), {});
  } else if (A.layout() == Layout::SparseCsr || A.layout() == Layout::SparseBsr) {
    // no broadcasting for non-strided layout
    set_output_raw_strided(0, self.sizes(), {}, self.options(), {}); // make row major strides for Sparse BLAS
    set_output_raw_strided(1, {0}, {}, self.options(), {}); // return 0-sized tensor
  } else if (A.layout() == Layout::SparseCsc) {
      TORCH_CHECK_VALUE(false, "triangular_solve: unsupported sparse layout.");
  } else {
    TORCH_INTERNAL_ASSERT(false, "triangular_solve: Got an unexpected layout.");
  }
}

TORCH_META_FUNC(_linalg_solve_ex)(const Tensor& A,
                                  const Tensor& B,
                                  bool left,
                                  bool check_errors) {
  // dtype
  at::native::checkFloatingOrComplex(A, "linalg.solve");
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "linalg.solve: Expected A and B to have the same dtype, but found A of type ",
              A.scalar_type(), " and B of type ", B.scalar_type(), " instead");

  // NumPy compat: Two types of 'B' tensors are supported:
  // - 1D tensor or batch of 1D tensors (vector case)
  // - 2D tensor or batch of 2D tensors (matrix case)
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(A, B);
  auto B_ = vector_case ? B.unsqueeze(-1) : B;

  // matrix shapes
  at::native::checkInputsSolver(A, B_, /*left=*/left, "linalg.solve");

  // Check that B can be broadcasted to the shape of A
  auto B_broad_shape = std::get<0>(at::native::_linalg_broadcast_batch_dims(B_, A));
  // We disallow the broadcasting of B as a vector when left=False as, in that case, A.shape = (*, 1, 1)
  TORCH_CHECK(left || !vector_case, "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. In this case linalg.solve is equivalent to B / A.squeeze(-1)");
  auto result_shape = vector_case ? IntArrayRef(B_broad_shape.data(), B_broad_shape.size() - 1)
                                  : B_broad_shape;
  // row major for mps implementation
  auto result_strides = at::native::batched_matrix_contiguous_strides(result_shape, /*f_contig=*/A.device().type() != at::kMPS? left : false);

  set_output_strided(0, result_shape, result_strides, B.options(), {});

  auto shape = A.sizes();
  auto ndim = shape.size();

  // LU, row major for mps
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/A.device().type() != at::kMPS? true : false);
  set_output_strided(1, shape, LU_strides, A.options(), {});

  // pivots
  set_output_contiguous(2, shape.slice(0, ndim - 1), A.options().dtype(kInt));

  // info
  set_output_contiguous(3, shape.slice(0, ndim - 2), A.options().dtype(kInt));
}

TORCH_META_FUNC(linalg_inv_ex)(const Tensor& A, bool check_errors) {
  at::native::squareCheckInputs(A, "linalg.inv");
  at::native::checkFloatingOrComplex(A, "linalg.inv", /*allow_low_precision_dtypes*/false);

  auto shape = A.sizes();

  auto result_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(0, shape, result_strides, A.options(), {});
  set_output_contiguous(
      1, shape.slice(0, shape.size() - 2), A.options().dtype(ScalarType::Int)); // info
}

TORCH_META_FUNC(linalg_lu_factor_ex)(const Tensor& A, bool pivot, bool check_errors) {
  TORCH_CHECK(A.dim() >= 2, "torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];

  // row major for MPS device, otherwise column major strides for BLAS
  auto LU_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/A.device().type() != at::kMPS);
  set_output_strided(0, sizes, LU_strides, A.options(), {});

  // Set sizes to the size of pivots
  sizes.pop_back();
  sizes.back() = std::min(m, n);
  set_output_contiguous(1, sizes, A.options().dtype(kInt), {});

  // Set sizes to the size of info
  sizes.pop_back();
  set_output_contiguous(2, sizes, A.options().dtype(kInt), {});
}

TORCH_META_FUNC(linalg_lu_solve)(const Tensor& LU,
                                 const Tensor& pivots,
                                 const Tensor& B,
                                 bool left,
                                 bool adjoint) {
  // dtype
  at::native::checkFloatingOrComplex(LU, "torch.linalg.lu_solve");
  TORCH_CHECK(LU.scalar_type() == B.scalar_type(),
              "linalg.lu_solve: Expected LU and B to have the same dtype, but found LU of type ",
              LU.scalar_type(), " and B of type ", B.scalar_type(), " instead");
  TORCH_CHECK(pivots.dtype() == at::kInt,
              "linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32");

  // matrix shapes
  at::native::squareCheckInputs(LU, "torch.linalg.lu_solve");
  at::native::checkInputsSolver(LU, B, left, "linalg.lu_solve");
  //
  TORCH_CHECK(LU.size(-1) == pivots.size(-1),
              "linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix");

  // batches
  TORCH_CHECK(
      LU.sizes().slice(0, LU.dim() - 1).equals(pivots.sizes()),
      "linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape ",
      pivots.sizes(), " instead");

  // This one checks that B can be broadcasted to the shape of A
  auto B_broadcast_size = std::get<0>(at::native::_linalg_broadcast_batch_dims(B, LU));
  auto result_strides = at::native::batched_matrix_contiguous_strides(B_broadcast_size, /*f_contig=*/left);

  set_output_strided(0, B_broadcast_size, result_strides, B.options(), {});
}

TORCH_META_FUNC(linalg_cholesky_ex)(const Tensor& A,
                                    bool upper,
                                    bool check_errors) {
  at::native::squareCheckInputs(A, "linalg.cholesky");
  at::native::checkFloatingOrComplex(A, "linalg.cholesky");

  auto A_shape = A.sizes();
  auto ndim = A_shape.size();

  // L
  auto L_strides = at::native::batched_matrix_contiguous_strides(A_shape, /*f-contig*=*/true);
  set_output_strided(0, A_shape, L_strides, A.options(), {});

  // info
  set_output_contiguous(1, A_shape.slice(0, ndim - 2), A.options().dtype(ScalarType::Int));
}

TORCH_META_FUNC(linalg_qr)(const Tensor& A,
                           std::string_view mode) {
  at::native::checkIsMatrix(A, "linalg.qr");
  at::native::checkFloatingOrComplex(A, "linalg.qr");
  auto [compute_q, reduced_mode] = at::native::_parse_qr_mode(mode);

  auto A_shape = A.sizes().vec();
  const auto m = A_shape.cend()[-2];
  const auto n = A_shape.cend()[-1];
  const auto k = std::min(m, n);

  if (compute_q) {
    auto Q_shape = A_shape;
    Q_shape.end()[-1] = reduced_mode ? k : m;
    auto Q_strides = at::native::batched_matrix_contiguous_strides(Q_shape, /*f-contig*=*/true);
    set_output_strided(0, Q_shape, Q_strides, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
  }

  // For readability
  auto R_shape = std::move(A_shape);
  R_shape.end()[-2] = (reduced_mode || !compute_q) ? k : m;
  auto R_strides = at::native::batched_matrix_contiguous_strides(R_shape, /*f-contig*=*/true);
  set_output_strided(1, R_shape, R_strides, A.options(), {});
}


TORCH_META_FUNC(_linalg_svd)(const Tensor& A,
                             bool full_matrices,
                             bool compute_uv,
                             std::optional<std::string_view> driver) {
  at::native::checkIsMatrix(A, "linalg.svd");
  at::native::checkFloatingOrComplex(A, "linalg.svd");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // Prepare sizes for U
  if (compute_uv) {
    sizes.back() = full_matrices ? m : k;
    auto U_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
    set_output_strided(0, sizes, U_strides, A.options(), {});

    // Prepare sizes for Vh
    sizes.end()[-2] = full_matrices ? n : k;
    sizes.end()[-1] = n;

    // We need to distinguish the cuSOLVER case, as the cuSOLVER algorithms we use
    // expect F-contig matrices, but they compute V rather than Vh
    const bool use_cusolver = at::native::svd_uses_cusolver(A);
    auto Vh_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/!use_cusolver);
    set_output_strided(2, sizes, Vh_strides, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
    set_output_raw_strided(2, {0}, {}, A.options(), {});
  }

  // Prepare sizes for S. S is always real, even when A is complex.
  sizes.pop_back();
  sizes.end()[-1] = k;
  set_output_contiguous(1, sizes, A.options().dtype(c10::toRealValueType(A.scalar_type())), {});
}

TORCH_META_FUNC(lu_unpack)(const Tensor& LU, const Tensor& pivots, bool unpack_data, bool unpack_pivots) {
  TORCH_CHECK(LU.dim() >= 2, "torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: ", LU.sizes(), " instead");
  if (unpack_pivots) {
    TORCH_CHECK(pivots.scalar_type() == at::kInt,
        "torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\n"
        "Note: this function is intended to be used with the output produced by torch.linalg.lu_factor");
  }

  auto sizes = LU.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // P.shape[-2:] == (m, m) (or size zero if pivot == False)
  sizes.end()[-1] = m;
  if (unpack_pivots) {
    set_output_raw_strided(0, sizes, {}, LU.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, LU.options(), {});
  }

  if (unpack_data) {
    // L.shape[-2:] == (m, k)
    sizes.end()[-1] = k;
    set_output_raw_strided(1, sizes, {}, LU.options(), {});

    // U.shape[-2:] == (k, n)
    sizes.end()[-2] = k;
    sizes.end()[-1] = n;
    set_output_raw_strided(2, sizes, {}, LU.options(), {});
  } else {
    set_output_raw_strided(1, {0}, {}, LU.options(), {});
    set_output_raw_strided(2, {0}, {}, LU.options(), {});
  }
}

TORCH_META_FUNC(_linalg_eigh)(const Tensor& A,
                              std::string_view uplo,
                              bool compute_v) {
  at::native::squareCheckInputs(A, "linalg.eigh");
  at::native::checkUplo(uplo);

  auto shape = A.sizes().vec();
  if (compute_v) {
    // eigenvectors
    auto V_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
    set_output_strided(1, shape, V_strides, A.options(), {});
  } else {
    set_output_raw_strided(1, {0}, {}, A.options(), {});
  }

  // eigenvalues
  shape.pop_back();
  set_output_contiguous(0, shape, A.options().dtype(c10::toRealValueType(A.scalar_type())), {});
}

TORCH_META_FUNC(linalg_lu)(const Tensor& A, bool pivot) {
  TORCH_CHECK(A.dim() >= 2, "linalg.lu: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // P.shape[-2:] == (m, m) (or size zero if pivot == False)
  sizes.end()[-1] = m;
  if (pivot) {
    set_output_raw_strided(0, sizes, {}, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
  }

  // L.shape[-2:] == (m, k)
  sizes.end()[-1] = k;
  set_output_raw_strided(1, sizes, {}, A.options(), {});

  // U.shape[-2:] == (k, n)
  sizes.end()[-2] = k;
  sizes.end()[-1] = n;
  set_output_raw_strided(2, sizes, {}, A.options(), {});
}

} // namespace at::meta

namespace at::native {

#if AT_BUILD_WITH_LAPACK()
// Define the per-batch functions to be used in the main implementation of the batched
// linear algebra operations

template<class scalar_t>
static void lapackCholeskySolve(char uplo, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, int *info);

template<> void lapackLu<c10::complex<double>>(int m, int n, c10::complex<double> *a, int lda, int *ipiv, int *info) {
  zgetrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, info);
}

template<> void lapackLu<c10::complex<float>>(int m, int n, c10::complex<float> *a, int lda, int *ipiv, int *info) {
  cgetrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, info);
}

template<> void lapackLu<double>(int m, int n, double *a, int lda, int *ipiv, int *info) {
  dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackLu<float>(int m, int n, float *a, int lda, int *ipiv, int *info) {
  sgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackCholeskySolve<c10::complex<double>>(char uplo, int n, int nrhs, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb, int *info) {
  zpotrs_(&uplo, &n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}

template<> void lapackCholeskySolve<c10::complex<float>>(char uplo, int n, int nrhs, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb, int *info) {
  cpotrs_(&uplo, &n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

template<> void lapackCholeskySolve<double>(char uplo, int n, int nrhs, double *a, int lda, double *b, int ldb, int *info) {
  dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholeskySolve<float>(char uplo, int n, int nrhs, float *a, int lda, float *b, int ldb, int *info) {
  spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholesky<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  zpotrf_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholesky<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  cpotrf_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholesky<double>(char uplo, int n, double *a, int lda, int *info) {
  dpotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholesky<float>(char uplo, int n, float *a, int lda, int *info) {
  spotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  zpotri_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  cpotri_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<double>(char uplo, int n, double *a, int lda, int *info) {
  dpotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<float>(char uplo, int n, float *a, int lda, int *info) {
  spotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackGeqrf<c10::complex<double>>(int m, int n, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *work, int lwork, int *info) {
  zgeqrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackGeqrf<c10::complex<float>>(int m, int n, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *work, int lwork, int *info) {
  cgeqrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackGeqrf<double>(int m, int n, double *a, int lda, double *tau, double *work, int lwork, int *info) {
  dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template<> void lapackGeqrf<float>(int m, int n, float *a, int lda, float *tau, float *work, int lwork, int *info) {
  sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrgqr<c10::complex<double>>(int m, int n, int k, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *work, int lwork, int *info) {
  zungqr_(&m, &n, &k, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackOrgqr<c10::complex<float>>(int m, int n, int k, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *work, int lwork, int *info) {
  cungqr_(&m, &n, &k, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackOrgqr<double>(int m, int n, int k, double *a, int lda, double *tau, double *work, int lwork, int *info) {
  dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrgqr<float>(int m, int n, int k, float *a, int lda, float *tau, float *work, int lwork, int *info) {
  sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrmqr<c10::complex<double>>(char side, char trans, int m, int n, int k, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *c, int ldc, c10::complex<double> *work, int lwork, int *info) {
  zunmqr_(&side, &trans, &m, &n, &k, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(c), &ldc, reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackOrmqr<c10::complex<float>>(char side, char trans, int m, int n, int k, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *c, int ldc, c10::complex<float> *work, int lwork, int *info) {
  cunmqr_(&side, &trans, &m, &n, &k, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(c), &ldc, reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackOrmqr<double>(char side, char trans, int m, int n, int k, double *a, int lda, double *tau, double *c, int ldc, double *work, int lwork, int *info) {
  dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
}

template<> void lapackOrmqr<float>(char side, char trans, int m, int n, int k, float *a, int lda, float *tau, float *c, int ldc, float *work, int lwork, int *info) {
  sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
}

template<> void lapackSyevd<c10::complex<double>, double>(char jobz, char uplo, int n, c10::complex<double> *a, int lda, double *w, c10::complex<double> *work, int lwork, double *rwork, int lrwork, int *iwork, int liwork, int *info) {
  zheevd_(&jobz, &uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, w, reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template<> void lapackSyevd<c10::complex<float>, float>(char jobz, char uplo, int n, c10::complex<float> *a, int lda, float *w, c10::complex<float> *work, int lwork, float *rwork, int lrwork, int *iwork, int liwork, int *info) {
  cheevd_(&jobz, &uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, w, reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template<> void lapackSyevd<double>(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork, double *rwork, int lrwork, int *iwork, int liwork, int *info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template<> void lapackSyevd<float>(char jobz, char uplo, int n, float *a, int lda, float *w, float *work, int lwork, float *rwork, int lrwork, int *iwork, int liwork, int *info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template<> void lapackEig<double>(char jobvl, char jobvr, int n, double *a, int lda, double *w, double* vl, int ldvl, double *vr, int ldvr, double *work, int lwork, double *rwork, int *info) {
  // lapack [sd]geev wants to separate output arrays: wr and wi for the real
  // and imaginary parts
  double *wr = w;
  double *wi = w ? w + n : nullptr;
  (void)rwork; // unused
  dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

template<> void lapackEig<float>(char jobvl, char jobvr, int n, float *a, int lda, float *w, float* vl, int ldvl, float *vr, int ldvr, float *work, int lwork, float *rwork, int *info) {
  // lapack [sd]geev wants to separate output arrays: wr and wi for the real
  // and imaginary parts
  float *wr = w;
  float *wi = w ? w  + n : nullptr;
  (void)rwork; // unused
  sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

template<> void lapackEig<c10::complex<double>, double>(char jobvl, char jobvr, int n, c10::complex<double> *a, int lda, c10::complex<double> *w, c10::complex<double> *vl, int ldvl, c10::complex<double> *vr, int ldvr, c10::complex<double> *work, int lwork, double *rwork, int *info) {
  zgeev_(&jobvl, &jobvr, &n,
         reinterpret_cast<std::complex<double>*>(a), &lda,
         reinterpret_cast<std::complex<double>*>(w),
         reinterpret_cast<std::complex<double>*>(vl), &ldvl,
         reinterpret_cast<std::complex<double>*>(vr), &ldvr,
         reinterpret_cast<std::complex<double>*>(work), &lwork,
         rwork, info);
}

template<> void lapackEig<c10::complex<float>, float>(char jobvl, char jobvr, int n, c10::complex<float> *a, int lda, c10::complex<float> *w, c10::complex<float> *vl, int ldvl, c10::complex<float> *vr, int ldvr, c10::complex<float> *work, int lwork, float *rwork, int *info) {
  cgeev_(&jobvl, &jobvr, &n,
         reinterpret_cast<std::complex<float>*>(a), &lda,
         reinterpret_cast<std::complex<float>*>(w),
         reinterpret_cast<std::complex<float>*>(vl), &ldvl,
         reinterpret_cast<std::complex<float>*>(vr), &ldvr,
         reinterpret_cast<std::complex<float>*>(work), &lwork,
         rwork, info);
}

template<> void lapackSvd<c10::complex<double>, double>(char jobz, int m, int n, c10::complex<double> *a, int lda,
                                  double *s, c10::complex<double> *u, int ldu, c10::complex<double> *vt, int ldvt, c10::complex<double> *work, int lwork, double *rwork, int *iwork, int *info) {
  zgesdd_(&jobz, &m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, s, reinterpret_cast<std::complex<double>*>(u), &ldu,
          reinterpret_cast<std::complex<double>*>(vt), &ldvt, reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, iwork, info);
}

template<> void lapackSvd<c10::complex<float>, float>(char jobz, int m, int n, c10::complex<float> *a, int lda,
                                 float *s, c10::complex<float> *u, int ldu, c10::complex<float> *vt, int ldvt, c10::complex<float> *work, int lwork, float *rwork, int *iwork, int *info) {
  cgesdd_(&jobz, &m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, s, reinterpret_cast<std::complex<float>*>(u), &ldu,
          reinterpret_cast<std::complex<float>*>(vt), &ldvt, reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, iwork, info);
}

template<> void lapackSvd<double>(char jobz, int m, int n, double *a, int lda,
                                  double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork, double *rwork, int *iwork, int *info) {
  dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
}

template<> void lapackSvd<float>(char jobz, int m, int n, float *a, int lda,
                                 float *s, float *u, int ldu, float *vt, int ldvt, float *work, int lwork, float *rwork, int *iwork, int *info) {
  sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
}

template <>
void lapackLdlSymmetric<double>(
    char uplo,
    int n,
    double* a,
    int lda,
    int* ipiv,
    double* work,
    int lwork,
    int* info) {
  dsytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlSymmetric<float>(
    char uplo,
    int n,
    float* a,
    int lda,
    int* ipiv,
    float* work,
    int lwork,
    int* info) {
  ssytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlSymmetric<c10::complex<double>>(
    char uplo,
    int n,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  zsytrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlSymmetric<c10::complex<float>>(
    char uplo,
    int n,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  csytrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlHermitian<double>(
    char uplo,
    int n,
    double* a,
    int lda,
    int* ipiv,
    double* work,
    int lwork,
    int* info) {
  dsytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlHermitian<float>(
    char uplo,
    int n,
    float* a,
    int lda,
    int* ipiv,
    float* work,
    int lwork,
    int* info) {
  ssytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlHermitian<c10::complex<double>>(
    char uplo,
    int n,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  zhetrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlHermitian<c10::complex<float>>(
    char uplo,
    int n,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  chetrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlSolveSymmetric<double>(
    char uplo,
    int n,
    int nrhs,
    double* a,
    int lda,
    int* ipiv,
    double* b,
    int ldb,
    int* info) {
  dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveSymmetric<float>(
    char uplo,
    int n,
    int nrhs,
    float* a,
    int lda,
    int* ipiv,
    float* b,
    int ldb,
    int* info) {
  ssytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveSymmetric<c10::complex<double>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* b,
    int ldb,
    int* info) {
  zsytrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveSymmetric<c10::complex<float>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* b,
    int ldb,
    int* info) {
  csytrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveHermitian<double>(
    char uplo,
    int n,
    int nrhs,
    double* a,
    int lda,
    int* ipiv,
    double* b,
    int ldb,
    int* info) {
  dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveHermitian<float>(
    char uplo,
    int n,
    int nrhs,
    float* a,
    int lda,
    int* ipiv,
    float* b,
    int ldb,
    int* info) {
  ssytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveHermitian<c10::complex<double>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* b,
    int ldb,
    int* info) {
  zhetrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveHermitian<c10::complex<float>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* b,
    int ldb,
    int* info) {
  chetrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(b),
      &ldb,
      info);
}

template<> void lapackLuSolve<c10::complex<double>>(char trans, int n, int nrhs, c10::complex<double> *a, int lda, int *ipiv, c10::complex<double> *b, int ldb, int *info) {
  zgetrs_(&trans, &n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}

template<> void lapackLuSolve<c10::complex<float>>(char trans, int n, int nrhs, c10::complex<float> *a, int lda, int *ipiv, c10::complex<float> *b, int ldb, int *info) {
  cgetrs_(&trans, &n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

template<> void lapackLuSolve<double>(char trans, int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int *info) {
  dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackLuSolve<float>(char trans, int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int *info) {
  sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGels<c10::complex<double>>(
    char trans, int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    c10::complex<double> *work, int lwork, int *info) {
  zgels_(&trans, &m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackGels<c10::complex<float>>(
    char trans, int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    c10::complex<float> *work, int lwork, int *info) {
  cgels_(&trans, &m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackGels<double>(
    char trans, int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *work, int lwork, int *info) {
  dgels_(&trans, &m, &n, &nrhs,
      a, &lda, b, &ldb, work, &lwork, info);
}

template<> void lapackGels<float>(
    char trans, int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *work, int lwork, int *info) {
  sgels_(&trans, &m, &n, &nrhs,
      a, &lda, b, &ldb, work, &lwork, info);
}

template<> void lapackGelsd<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    double *s, double rcond, int *rank,
    c10::complex<double> *work, int lwork,
    double *rwork, int *iwork, int *info) {
  zgelsd_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, iwork, info);
}

template<> void lapackGelsd<c10::complex<float>, float>(
    int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    float *s, float rcond, int *rank,
    c10::complex<float> *work, int lwork,
    float *rwork, int *iwork, int *info) {
  cgelsd_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<float>*>(work), &lwork,
      rwork, iwork, info);
}

template<> void lapackGelsd<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *s, double rcond, int *rank,
    double *work, int lwork,
    double *rwork, int *iwork, int *info) {
  dgelsd_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, iwork, info);
}

template<> void lapackGelsd<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *s, float rcond, int *rank,
    float *work, int lwork,
    float *rwork, int *iwork, int *info) {
  sgelsd_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, iwork, info);
}

template<> void lapackGelsy<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    int *jpvt, double rcond, int *rank,
    c10::complex<double> *work, int lwork, double *rwork, int *info) {
  zgelsy_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      jpvt, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelsy<c10::complex<float>, float>(
    int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    int *jpvt, float rcond, int *rank,
    c10::complex<float> *work, int lwork, float *rwork, int *info) {
  cgelsy_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      jpvt, &rcond, rank,
      reinterpret_cast<std::complex<float>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelsy<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    int *jpvt, double rcond, int *rank,
    double *work, int lwork, double *rwork, int *info) {
  dgelsy_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      jpvt, &rcond, rank,
      work, &lwork, info);
}

template<> void lapackGelsy<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    int *jpvt, float rcond, int *rank,
    float *work, int lwork, float *rwork, int *info) {
  sgelsy_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      jpvt, &rcond, rank,
      work, &lwork, info);
}

template<> void lapackGelss<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    double *s, double rcond, int *rank,
    c10::complex<double> *work, int lwork,
    double *rwork, int *info
    ) {
  zgelss_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelss<c10::complex<float>, float>(
    int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    float *s, float rcond, int *rank,
    c10::complex<float> *work, int lwork,
    float *rwork, int *info
    ) {
  cgelss_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<float>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelss<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *s, double rcond, int *rank,
    double *work, int lwork,
    double *rwork, int *info) {
  dgelss_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, info);
}

template<> void lapackGelss<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *s, float rcond, int *rank,
    float *work, int lwork,
    float *rwork, int *info) {
  sgelss_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, info);
}
#endif

#if AT_BUILD_WITH_BLAS()
template<> void blasTriangularSolve<c10::complex<double>>(char side, char uplo, char trans, char diag, int n, int nrhs, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb) {
  std::complex<double> one{1., 0.};
  ztrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(b), &ldb);
}

template<> void blasTriangularSolve<c10::complex<float>>(char side, char uplo, char trans, char diag, int n, int nrhs, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb) {
  std::complex<float> one{1.f, 0.f};
  ctrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(b), &ldb);
}

template<> void blasTriangularSolve<double>(char side, char uplo, char trans, char diag, int n, int nrhs, double *a, int lda, double *b, int ldb) {
  auto one = 1.;
  dtrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, a, &lda, b, &ldb);
}

template<> void blasTriangularSolve<float>(char side, char uplo, char trans, char diag, int n, int nrhs, float *a, int lda, float *b, int ldb) {
  auto one = 1.f;
  strsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, a, &lda, b, &ldb);
}
#endif

void _linalg_check_errors(
    const Tensor& infos,
    const std::string_view api_name,
    bool is_matrix) {
  TORCH_INTERNAL_ASSERT(infos.scalar_type() == kInt);
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());
  if (infos.is_meta()) {
    return;
  }

  // If it's all zeros, we return early.
  // We optimise for the most likely case.
  if (C10_LIKELY(!infos.any().item<bool>())) {
    return;
  }

  int32_t info = 0;
  std::string batch_str;
  if (is_matrix) {
    info = infos.item<int>();
    // batch_str needn't be set for matrices
  } else {
    // Find the first non-zero info
    auto infos_cpu = infos.to(at::kCPU);
    auto ptr = infos_cpu.const_data_ptr<int32_t>();
    auto n = infos.numel();
    auto info_ptr = std::find_if(ptr, ptr + n, [](int32_t x) { return x != 0; });
    info = *info_ptr;
    batch_str = ": (Batch element " + std::to_string(std::distance(ptr, info_ptr)) + ")";
  }

  if (info < 0) {
    // Reference LAPACK 3.10+ changed `info` behavior for inputs with non-finite values
    // Previously, it would return `info` > 0, but now it returns `info` = -4
    // OpenBLAS 0.3.15+ uses the Reference LAPACK 3.10+.
    // MKL 2022.0+ uses the Reference LAPACK 3.10+.
    // Older version of MKL and OpenBLAS follow the old behavior (return `info` > 0).
    // Here we check for the case where `info` is -4 and raise an error
    if (api_name.find("svd") != api_name.npos) {
      TORCH_CHECK_LINALG(info != -4, api_name, batch_str,
          ": The algorithm failed to converge because the input matrix contained non-finite values.");
    }
    TORCH_INTERNAL_ASSERT(false, api_name, batch_str,
        ": Argument ", -info, " has illegal value. Most certainly there is a bug in the implementation calling the backend library.");
  } else if (info > 0) {
    if (api_name.find("inv") != api_name.npos) {
      // inv, inverse, cholesky_inverse, etc.
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The diagonal element ", info, " is zero, the inversion could not be completed because the input matrix is singular.");
    } else if (api_name.find("solve") != api_name.npos) {
      // solve, linalg_solve, cholesky_solve, etc.
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The solver failed because the input matrix is singular.");
    } else if (api_name.find("cholesky") != api_name.npos) {
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The factorization could not be completed because the input is not positive-definite (the leading minor of order ", info, " is not positive-definite).");
    } else if (api_name.find("svd") != api_name.npos) {
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: ", info, ").");
    } else if (api_name.find("eig") != api_name.npos || api_name.find("syevd") != api_name.npos) {
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: ", info, ").");
    } else if (api_name.find("lstsq") != api_name.npos) {
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The least squares solution could not be computed because the input matrix does not have full rank (error code: ", info, ").");
    } else if (api_name.find("lu_factor") != api_name.npos) {
      TORCH_CHECK(false, api_name, batch_str,
          ": U[", info, ",", info, "] is zero and using it on lu_solve would result in a division by zero. "
          "If you still want to perform the factorization, consider calling linalg.lu(A, pivot) or "
          "linalg.lu_factor_ex(A, pivot)");
    } else {
      TORCH_INTERNAL_ASSERT(false, api_name, ": Unknown error code: ", info, ".");
    }
  }
  // We should never reach this point as info was non-zero
  TORCH_INTERNAL_ASSERT(false);
}

// If an input requires fw or bw grad then we need to go down a different
// (slower) path to ensure that the gradients are computable.
// That is what `_may_require_fw_or_bw_grad` is helpful for.
//
// Why is there a isTensorSubclassLike check here?
// Without it, this function can lead to composite compliance problems, which
// may lead to bugs in functorch, where a Tensor Subclass that doesn't
// require grad may wrap a Tensor subclass that requires grad.
static bool _may_require_fw_or_bw_grad(const Tensor& input) {
  return ((at::GradMode::is_enabled() && input.requires_grad())
          || input._fw_grad(/*level */ 0).defined()
          || isTensorSubclassLike(input));
}

// NOLINTBEGIN(cppcoreguidelines-pro-type-const-cast)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.inv ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TORCH_IMPL_FUNC(linalg_inv_ex_out)(const Tensor& A, bool check_errors, const Tensor& result, const Tensor& info) {
  // Fill result with the identity
  result.zero_();
  result.diagonal(0, -2, -1).fill_(1.);
  at::linalg_solve_ex_out(const_cast<Tensor&>(result), const_cast<Tensor&>(info), A, result, /*left*/true);
  if (check_errors) {
    at::_linalg_check_errors(info, "linalg.inv_ex", A.dim() == 2);
  }
}

Tensor& linalg_inv_out(const Tensor& A, Tensor& result) {
  auto info = at::empty({0}, A.options().dtype(kInt));
  at::linalg_inv_ex_out(result, info, A);
  at::_linalg_check_errors(info, "linalg.inv", A.dim() == 2);
  return result;
}

Tensor linalg_inv(const Tensor& A) {
  auto [result, info] = at::linalg_inv_ex(A);
  at::_linalg_check_errors(info, "linalg.inv", A.dim() == 2);
  return result;
}

Tensor& inverse_out(const Tensor& A, Tensor& result) {
  return at::linalg_inv_out(result, A);
}

Tensor inverse(const Tensor& A) {
  return at::linalg_inv(A);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, Tensor& infos) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "cholesky_solve: LAPACK library not found in compilation");
#else
  char uplo = upper ? 'U' : 'L';

  auto A_data = A.const_data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);
  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto ldab = std::max<int64_t>(1, n);
  auto nrhs = b.size(-1);

  for (const auto i : c10::irange(batch_size)) {
    const scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
    int info = 0;
    lapackCholeskySolve<scalar_t>(uplo, n, nrhs, const_cast<scalar_t*>(A_working_ptr), ldab, b_working_ptr, ldab, &info);
    infos_data[i] = info;
    if (info != 0) {
      return;
    }
  }
#endif
}

Tensor _cholesky_solve_helper_cpu(const Tensor& self, const Tensor& A, bool upper) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  auto infos = at::zeros({batchCount(self)}, self.options().dtype(kInt));
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_solve_cpu", [&]{
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, infos);
  });

  at::_linalg_check_errors(infos, "cholesky_solve_cpu", self.dim() == 2);
  return self_working_copy;
}

// Supports arbitrary batch dimensions for self and A
Tensor cholesky_solve(const Tensor& self, const Tensor& A, bool upper) {
  TORCH_CHECK(self.dim() >= 2,
           "b should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  TORCH_CHECK(A.dim() >= 2,
           "u should have at least 2 dimensions, but has ", A.dim(), " dimensions instead");
  auto [self_broadcasted, A_broadcasted] = _linalg_broadcast_batch_dims(self, A, "cholesky_solve");
  return at::_cholesky_solve_helper(self_broadcasted, A_broadcasted, upper);
}

Tensor& cholesky_solve_out(const Tensor& self, const Tensor& A, bool upper, Tensor& result) {
  checkSameDevice("cholesky_solve", result, self);
  checkLinalgCompatibleDtype("cholesky_solve", result, self);
  Tensor result_tmp = at::cholesky_solve(self, A, upper);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(cholesky_stub);

Tensor cholesky(const Tensor &self, bool upper) {
   TORCH_WARN_ONCE(
    "torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be ",
    "removed in a future PyTorch release.\n",
    "L = torch.cholesky(A)\n",
    "should be replaced with\n",
    "L = torch.linalg.cholesky(A)\n",
    "and\n"
    "U = torch.cholesky(A, upper=True)\n",
    "should be replaced with\n",
    "U = torch.linalg.cholesky(A).mH\n"
    "This transform will produce equivalent results for all valid (symmetric positive definite) inputs."
  );
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  squareCheckInputs(self, "cholesky");

  auto raw_cholesky_output = cloneBatchedColumnMajor(self);
  auto info_shape = IntArrayRef(
      self.sizes().cbegin(), self.sizes().cend() - 2); // self.shape[:-2]
  auto info = at::empty({info_shape}, self.options().dtype(kInt));

  // fill the raw_cholesky_output with the result
  cholesky_stub(self.device().type(), raw_cholesky_output, info, upper);

  at::_linalg_check_errors(info, "cholesky", self.dim() == 2);

  if (upper) {
    return raw_cholesky_output.triu_();
  } else {
    return raw_cholesky_output.tril_();
  }
}

Tensor& cholesky_out(const Tensor &self, bool upper, Tensor &result) {
   TORCH_WARN_ONCE(
    "torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be ",
    "removed in a future PyTorch release.\n",
    "L = torch.cholesky(A)\n",
    "should be replaced with\n",
    "L = torch.linalg.cholesky(A)\n",
    "and\n"
    "U = torch.cholesky(A, upper=True)\n",
    "should be replaced with\n",
    "U = torch.linalg.cholesky(A).mH\n"
    "This transform will produce equivalent results for all valid (symmetric positive definite) inputs."
  );
  checkSameDevice("cholesky", result, self);
  checkLinalgCompatibleDtype("cholesky", result, self);
  Tensor result_tmp = at::cholesky(self, upper);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

TORCH_IMPL_FUNC(linalg_cholesky_ex_out)(const Tensor& A,
                                        bool upper,
                                        bool check_errors,
                                        const Tensor& L,
                                        const Tensor& info) {
  // Nothing to do there
  if (L.numel() == 0) {
    info.zero_();
    return;
  }
  const auto cpu = A.device() == kCPU;

  // We can perform this optimisation just on CPU as it fails for MAGMA
  // due to some bug
  if (cpu) {
    if (upper) {
      at::triu_out(const_cast<Tensor&>(L), A);
    } else {
      at::tril_out(const_cast<Tensor&>(L), A);
    }
  } else {
    L.copy_(A);
  }

  cholesky_stub(L.device().type(), L, info, upper);

  if (!cpu) {
    if (upper) {
      L.triu_();
    } else {
      L.tril_();
    }
  }

  if (check_errors) {
    at::_linalg_check_errors(info, "linalg.cholesky_ex", A.dim() == 2);
  }
}

Tensor linalg_cholesky(const Tensor& A, bool upper) {
  auto [L, info] = at::linalg_cholesky_ex(A, upper, /*check_errors=*/false);
  at::_linalg_check_errors(info, "linalg.cholesky", A.dim() == 2);
  return L;
}

Tensor& linalg_cholesky_out(const Tensor& A, bool upper, Tensor& L) {
  auto info = at::empty({0}, A.options().dtype(kInt));
  at::linalg_cholesky_ex_out(L, info, A, upper, /*check_errors=*/false);
  at::_linalg_check_errors(info, "linalg.cholesky", A.dim() == 2);
  return L;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(cholesky_inverse_stub);

static Tensor& cholesky_inverse_out_info(Tensor& result, Tensor& infos, const Tensor& input, bool upper) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-1) == input.size(-2));

  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  TORCH_INTERNAL_ASSERT(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT(infos.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT(infos.numel() == std::max<int64_t>(1, batchCount(input)));

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, input.mT(), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // result tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(result.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT(result.sizes().equals(input.sizes()));

  // cholesky_inverse_stub (apply_cholesky_inverse) performs calculations in-place and result must be a copy of input
  result.copy_(input);

  // infos must be contiguous
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());
  infos.fill_(0);

  result = cholesky_inverse_stub(result.device().type(), result, infos, upper);
  return result;
}

Tensor& cholesky_inverse_out(const Tensor &input, bool upper, Tensor &result) {
  squareCheckInputs(input, "cholesky_inverse");
  checkSameDevice("cholesky_inverse", result, input);
  checkLinalgCompatibleDtype("cholesky_inverse", result, input);

  // MAGMA requires 'infos' to reside in CPU memory, therefore we create 'infos' only on CPU for now.
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, input.options().dtype(kInt).device(kCPU));

  bool result_input_same_type = (result.scalar_type() == input.scalar_type());
  bool result_equal_expected_shape = result.sizes().equals(input.sizes());
  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.mT().is_contiguous();
  }

  // if result is not empty and not in batched column major format
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= !result_input_same_type;  // or result does not have the same dtype as input
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape); // or result does not have the expected shape
  // we have to allocate a temporary tensor
  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, input.options());
    result_tmp = cholesky_inverse_out_info(result_tmp, infos, input, upper);
    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
  } else {
    // use result's memory directly
    result = cholesky_inverse_out_info(result, infos, input, upper);
  }

  // Now check LAPACK/MAGMA error codes
  at::_linalg_check_errors(infos, "cholesky_inverse", result.dim() == 2);
  return result;
}

Tensor cholesky_inverse(const Tensor &input, bool upper) {
  Tensor result = at::empty({0}, input.options());
  result = at::cholesky_inverse_out(result, input, upper);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Auxiliary function that returns the LU decomposition to use it in the backward
TORCH_IMPL_FUNC(_linalg_solve_ex_out)(const Tensor& A,
                                      const Tensor& B,
                                      bool left,
                                      bool check_errors,
                                      const Tensor& result,
                                      const Tensor& LU,
                                      const Tensor& pivots,
                                      const Tensor& info) {
  // Possible optimization: Compute the LU factorization of A^T if A is contiguous
  // Then we solve A^T X = B with adjoint=True
  // This saves a copy as A doesn't need to be copied into an F-contig matrix in lu_factor
  // This optimization makes functorch's batching rule difficult. See NOTE [ solve_ex Batch Rule Contiguity ]
  const bool use_A_T = A.is_contiguous() && !A.is_complex();
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU),
                              const_cast<Tensor&>(pivots),
                              const_cast<Tensor&>(info),
                              use_A_T ? A.mT() : A);
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.solve_ex", A.dim() == 2);
  }

  // [numpy-compat] Handle vectors on the rhs
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(LU, B);
  auto result_ = vector_case ? result.unsqueeze(-1) : result;
  auto B_ = vector_case ? B.unsqueeze(-1) : B;
  at::linalg_lu_solve_out(result_, LU, pivots, B_, left, /*adjoint*/use_A_T);
}

std::tuple<Tensor&, Tensor&> linalg_solve_ex_out(const Tensor& A,
                                                 const Tensor& B,
                                                 bool left,
                                                 bool check_errors,
                                                 Tensor& result,
                                                 Tensor& info) {
  auto LU = B.new_empty({0});
  auto pivots = B.new_empty({0}, kInt);
  at::_linalg_solve_ex_out(result, LU, pivots, info, A, B, left, check_errors);
  return std::tie(result, info);
}

// We implement linalg_solve_ex as a composite function of _linalg_solve
std::tuple<Tensor, Tensor> linalg_solve_ex(const Tensor& A,
                                           const Tensor& B,
                                           bool left,
                                           bool check_errors) {
  auto [result, LU, pivots, info] = at::_linalg_solve_ex(A, B, left, check_errors);
  return std::make_tuple(std::move(result), std::move(info));
}

Tensor& linalg_solve_out(const Tensor& A,
                         const Tensor& B,
                         bool left,
                         Tensor& result) {
  auto info = B.new_empty({0}, kInt);
  at::linalg_solve_ex_out(result, info, A, B, left);
  at::_linalg_check_errors(info, "torch.linalg.solve", A.dim() == 2);
  return result;
}

Tensor linalg_solve(const Tensor& A,
                    const Tensor& B,
                    bool left) {
  if (A.layout() == kSparseCsr) {
    return at::_spsolve(A, B, left);
  }
  auto [result, info] = at::linalg_solve_ex(A, B, left);
  at::_linalg_check_errors(info, "torch.linalg.solve", A.dim() == 2);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_factor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(lu_factor_stub);

TORCH_IMPL_FUNC(linalg_lu_factor_ex_out)(const Tensor& A,
                                         bool pivot,
                                         bool check_errors,
                                         const Tensor& LU,
                                         const Tensor& pivots,
                                         const Tensor& info) {
  if (A.numel() == 0) {
    // zero out the infos as it will have one element if the input is a matrix of size (0, 0)
    info.zero_();
    return;
  }
  if (!LU.is_same(A)) {
    LU.copy_(A);
  }

  lu_factor_stub(A.device().type(), LU, pivots, info, pivot);

  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.lu_factor_ex", A.dim() == 2);
  }
}

std::tuple<Tensor&, Tensor&> linalg_lu_factor_out(const Tensor& A, bool pivot, Tensor& LU, Tensor& pivots) {
  auto info = at::empty({0}, A.options().dtype(kInt));
  // We pass check_errors as we want to use lu_factor rather than lu_factor_ex in the errors
  at::linalg_lu_factor_ex_out(LU, pivots, info, A, pivot, /*check_errors=*/false);
  at::_linalg_check_errors(info, "torch.linalg.lu_factor", A.dim() == 2);
  return std::tie(LU, pivots);
}

std::tuple<Tensor, Tensor> linalg_lu_factor(const Tensor& A, bool pivot) {
  auto [LU, pivots, info] = at::linalg_lu_factor_ex(A, pivot, /*check_errors=*/false);
  at::_linalg_check_errors(info, "torch.linalg.lu_factor", A.dim() == 2);
  return std::make_tuple(std::move(LU), std::move(pivots));
}

// TODO Deprecate this function in favour of linalg_lu_factor_ex
std::tuple<Tensor, Tensor, Tensor> _lu_with_info(const Tensor& self, bool compute_pivots, bool /*unused*/) {
   TORCH_WARN_ONCE(
    "torch.lu is deprecated in favor of torch.linalg.lu_factor / torch.linalg.lu_factor_ex and will be ",
    "removed in a future PyTorch release.\n",
    "LU, pivots = torch.lu(A, compute_pivots)\n",
    "should be replaced with\n",
    "LU, pivots = torch.linalg.lu_factor(A, compute_pivots)\n",
    "and\n",
    "LU, pivots, info = torch.lu(A, compute_pivots, get_infos=True)\n",
    "should be replaced with\n",
    "LU, pivots, info = torch.linalg.lu_factor_ex(A, compute_pivots)"
  );
  return at::linalg_lu_factor_ex(self, compute_pivots, false);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(unpack_pivots_stub);

TORCH_IMPL_FUNC(linalg_lu_out)(const Tensor& A,
                               bool pivot,
                               const Tensor& P,
                               const Tensor& L,
                               const Tensor& U) {
  const auto m = A.sizes().end()[-2];
  const auto n = A.sizes().end()[-1];

  // A.shape[-2:] == (m, n)
  // P.shape[-2:] == (m, m)
  // L.shape[-2:] == (m, k)
  // U.shape[-2:] == (k, n)
  // with k = min(m, n)

  // Use L as it has the correct size
  const bool use_L = m > n;
  auto pivots = at::empty({0}, A.options().dtype(kInt));
  auto info = at::empty({0}, A.options().dtype(kInt));
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(use_L ? L : U),
                              const_cast<Tensor&>(pivots),
                              const_cast<Tensor&>(info),
                              A,
                              pivot,
                              /*check_errors=*/false);
  at::lu_unpack_out(const_cast<Tensor&>(P),
                    const_cast<Tensor&>(L),
                    const_cast<Tensor&>(U),
                    use_L ? L : U,
                    pivots,
                    /*unpack_data=*/true,
                    /*unpack_pivots=*/pivot);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_unpack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TORCH_IMPL_FUNC(lu_unpack_out)(const Tensor& LU,
                               const Tensor& pivots,
                               bool unpack_lu,
                               bool unpack_pivots,
                               const Tensor& P,
                               const Tensor& L,
                               const Tensor& U) {
  const auto m = LU.sizes().end()[-2];
  const auto n = LU.sizes().end()[-1];

  // A.shape[-2:] == (m, n)
  // P.shape[-2:] == (m, m)
  // L.shape[-2:] == (m, k)
  // U.shape[-2:] == (k, n)
  // with k = min(m, n)

  if (unpack_lu) {
    if (m > n || LU.is_same(L)) {
      // The order of triu and tril is important as we may have LU.is_same(L)
      at::triu_out(const_cast<Tensor&>(U), m == n ? LU : LU.narrow(-2, 0, n), 0);
      at::tril_out(const_cast<Tensor&>(L), LU, -1);
      L.diagonal(0, -2, -1).fill_(1.);
    } else {
      // The order of triu and tril is important as we may have LU.is_same(U)
      at::tril_out(const_cast<Tensor&>(L), m == n ? LU : LU.narrow(-1, 0, m), -1);
      L.diagonal(0, -2, -1).fill_(1.);
      at::triu_out(const_cast<Tensor&>(U), LU, 0);
    }
  }
  if (unpack_pivots) {
    // lu_factor_ex returns an int32 1-based indexing, which is what we have in `pivots`
    // We transform that to a proper permutation of the indices {0, ..., m-1}
    const auto perm_sizes = IntArrayRef(P.sizes().data(), P.dim() - 1);

    // Fill `perm` with the identity permutation (perhaps batched)
    const auto perm = at::arange(m, pivots.options().memory_format(at::MemoryFormat::Contiguous).dtype(kLong))
                        .expand(perm_sizes)
                        .contiguous();

    // Note that perm is of type kLong and pivots is a 1-indexed kInt.
    // This is taken into account in the unpack_pivots kernel
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .declare_static_shape(pivots.sizes(), /*squash_dims=*/pivots.dim() - 1)
      .add_output(perm)
      .add_owned_const_input(pivots.contiguous())
      .build();

    unpack_pivots_stub(pivots.device().type(), iter, std::min(m, n), m);

    // Transform the permutation into a permutation matrix
    P.zero_();
    P.scatter_(-2, perm.unsqueeze(-2), 1.);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DEFINE_DISPATCH(lu_solve_stub);

TORCH_IMPL_FUNC(linalg_lu_solve_out)(const Tensor& LU,
                                     const Tensor& pivots,
                                     const Tensor& B,
                                     bool left,
                                     bool adjoint,
                                     const Tensor& result) {
  // Trivial case
  if (result.numel() == 0) {
    return;
  }

  // Solve A^H X = B^H. Then we return X^H
  if (!left) {
    adjoint = !adjoint;
    result.transpose_(-2, -1);
  }

  // Copy B (or B^H) into result
  if (!result.is_same(B)) {
    result.copy_(left ? B : B.mH());
  }

  // Make LU / pivots F-contiguous
  auto pivots_ = pivots.expect_contiguous();
  auto LU_ = at::native::borrow_else_clone(
      LU.mT().is_contiguous(), LU, LU, /*contig=*/false);

  const auto trans = !adjoint ? TransposeType::NoTranspose :
                     LU.is_complex() ? TransposeType::ConjTranspose
                                     : TransposeType::Transpose;

  lu_solve_stub(LU_->device().type(), *LU_, *pivots_, result, trans);

  // Conj-transpose back in-place
  if (!left) {
    result.transpose_(-2, -1);
    if (result.is_complex()) {
      result._set_conj(!result.is_conj());
    }
  }
}

Tensor lu_solve(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots) {
  TORCH_WARN_ONCE(
    "torch.lu_solve is deprecated in favor of torch.linalg.lu_solve",
    "and will be removed in a future PyTorch release.\n",
    "Note that torch.linalg.lu_solve has its arguments reversed.\n",
    "X = torch.lu_solve(B, LU, pivots)\n",
    "should be replaced with\n",
    "X = torch.linalg.lu_solve(LU, pivots, B)"
  );
  return at::linalg_lu_solve(LU_data, LU_pivots, self);
}

Tensor& lu_solve_out(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots, Tensor& result) {
  TORCH_WARN_ONCE(
    "torch.lu_solve is deprecated in favor of torch.linalg.lu_solve",
    "and will be removed in a future PyTorch release.\n",
    "Note that torch.linalg.lu_solve has its arguments reversed.\n",
    "X = torch.lu_solve(B, LU, pivots)\n",
    "should be replaced with\n",
    "X = torch.linalg.lu_solve(LU, pivots, B)"
  );
  return at::linalg_lu_solve_out(result, LU_data, LU_pivots, self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(triangular_solve_stub);

/*
Solves the matrix equation 'input' @ 'result' = 'other' for the 'result'.
The result of the computation is saved in-place in 'result' tensor,
'clone_input' will be a copy of 'input',
'infos' is used to store information for possible checks for error,
'upper' controls the portion of input matrix to consider in computations,
'transpose' if true then 'input.mT()' @ 'result' = 'other' is solved,
'unitriangular' if true then the diagonal elements of 'input' are assumed to be 1
and the actual diagonal values are not used.
*/
static void triangular_solve_out_impl(
    const Tensor& result,
    const Tensor& clone_input,
    const Tensor& input,
    const Tensor& other,
    bool upper, bool transpose, bool unitriangular) {
  TORCH_WARN_ONCE(
    "torch.triangular_solve is deprecated in favor of torch.linalg.solve_triangular",
    "and will be removed in a future PyTorch release.\n",
    "torch.linalg.solve_triangular has its arguments reversed and does not return a copy of one of the inputs.\n",
    "X = torch.triangular_solve(B, A).solution\n",
    "should be replaced with\n",
    "X = torch.linalg.solve_triangular(A, B).");
  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of
  // the hierarchy of calls
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.size(-2) == input.size(-1));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == other.device());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == result.device());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == clone_input.device());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == other.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == result.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == clone_input.scalar_type());

  // if 'result' has no elements we can modify it
  if (result.numel() == 0) {
    result.resize_(other.mT().sizes(), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);  // make 'result' to have Fortran contiguous memory layout
  }

  // if 'clone_input' has no elements we can modify it
  if (clone_input.numel() == 0) {
    clone_input.resize_(input.mT().sizes(), MemoryFormat::Contiguous);
    clone_input.transpose_(-2, -1);  // make 'clone_input' to have Fortran contiguous memory layout
  }

  // 'result' and 'clone_input' must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(clone_input.mT().is_contiguous());

  // triangular_solve_stub performs calculations in-place
  // 'result' must be a copy of 'other'
  // 'clone_input' must be a copy of 'input'
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(other.sizes()));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(clone_input.sizes().equals(input.sizes()));
  result.copy_(other);
  clone_input.copy_(input);

  triangular_solve_stub(input.device().type(), clone_input, result, /*left=*/true, upper, transpose ? TransposeType::Transpose : TransposeType::NoTranspose, unitriangular);
}

TORCH_IMPL_FUNC(triangular_solve_out)(const Tensor& self, const Tensor& A, bool upper, bool transpose, bool unitriangular, const Tensor& result, const Tensor& clone_A) {
  auto [self_broadcast, A_broadcast] = _linalg_broadcast_batch_dims(self, A, "triangular_solve");

  bool copy_needed = !result.transpose(-2, -1).is_contiguous();
  copy_needed |= !clone_A.transpose(-2, -1).is_contiguous();

  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, self.options());
    Tensor clone_A_tmp = at::empty({0}, A.options());

    triangular_solve_out_impl(result_tmp, clone_A_tmp, A_broadcast, self_broadcast, upper, transpose, unitriangular);

    result.copy_(result_tmp);
    clone_A.copy_(clone_A_tmp);
  } else {
    triangular_solve_out_impl(result, clone_A, A_broadcast, self_broadcast, upper, transpose, unitriangular);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(geqrf_stub);

static void geqrf_out_helper(const Tensor& input, const Tensor& QR, const Tensor& tau) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);

  TORCH_INTERNAL_ASSERT(input.scalar_type() == QR.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == QR.device());

  TORCH_INTERNAL_ASSERT(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == tau.device());

  // if 'QR' has no elements we can modify it
  if (QR.numel() == 0) {
    QR.resize_as_(input.mT(), MemoryFormat::Contiguous);
    QR.transpose_(-2, -1); // make Fortran-contiguous
  }

  auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2).vec(); // input.shape[:-2]
  expected_batch_tau_shape.push_back(std::min(input.size(-2), input.size(-1)));
  if (tau.numel() == 0) {
    tau.resize_(expected_batch_tau_shape);
  }

  // QR tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(QR.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT(QR.sizes().equals(input.sizes()));

  // tau tensor must be contiguous
  TORCH_INTERNAL_ASSERT(tau.is_contiguous());
  TORCH_INTERNAL_ASSERT(tau.sizes().equals(expected_batch_tau_shape));

  // geqrf_stub (apply_geqrf) performs calculations in-place and 'QR' must be a copy of input
  QR.copy_(input);
  geqrf_stub(input.device().type(), QR, tau);
}

std::tuple<Tensor&, Tensor&> geqrf_out(const Tensor& input, Tensor& QR, Tensor& tau) {
  TORCH_CHECK(input.dim() >= 2, "torch.geqrf: input must have at least 2 dimensions.");

  checkSameDevice("torch.geqrf", QR, input, "a"); // 'a' is used in documentation and native_functions.yml
  checkSameDevice("torch.geqrf", tau, input, "tau");
  checkLinalgCompatibleDtype("torch.geqrf", QR, input, "a");
  checkLinalgCompatibleDtype("torch.geqrf", tau, input, "tau");

  bool QR_input_same_type = (QR.scalar_type() == input.scalar_type());
  bool tau_input_same_type = (tau.scalar_type() == input.scalar_type());
  bool QR_equal_expected_shape = QR.sizes().equals(input.sizes());

  auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2).vec(); // input.shape[:-2]
  expected_batch_tau_shape.push_back(std::min(input.size(-2), input.size(-1)));
  bool tau_equal_expected_shape = tau.sizes().equals(expected_batch_tau_shape);

  bool is_batched_column_major = false;
  if (QR.dim() >= 2) {
    is_batched_column_major = QR.mT().is_contiguous();
  }

  // if 'QR' is not empty and not in batched column major format
  bool copy_needed = (QR.numel() != 0 && !is_batched_column_major);
  copy_needed |= (QR.numel() != 0 && !QR_equal_expected_shape); // or 'QR' does not have the expected shape
  copy_needed |= !QR_input_same_type;  // or 'QR' does not have the same dtype as input
  // we have to allocate a temporary tensor

  copy_needed |= (tau.numel() != 0 && !tau.is_contiguous());
  copy_needed |= (tau.numel() != 0 && !tau_equal_expected_shape); // or 'tau' does not have the expected shape
  copy_needed |= !tau_input_same_type;  // or 'tau' does not have the same dtype as input

  if (copy_needed) {
    Tensor QR_tmp = at::empty({0}, input.options());
    Tensor tau_tmp = at::empty({0}, input.options());

    geqrf_out_helper(input, QR_tmp, tau_tmp);

    at::native::resize_output(QR, QR_tmp.sizes());
    QR.copy_(QR_tmp);
    at::native::resize_output(tau, tau_tmp.sizes());
    tau.copy_(tau_tmp);
  } else {
    // use "out" tensors' storage directly
    geqrf_out_helper(input, QR, tau);
  }

  return std::tuple<Tensor&, Tensor&>(QR, tau);
}

std::tuple<Tensor, Tensor> geqrf(const Tensor& input) {
  Tensor QR = at::empty({0}, input.options());
  Tensor tau = at::empty({0}, input.options());
  std::tie(QR, tau) = at::geqrf_outf(input, QR, tau);
  return std::make_tuple(std::move(QR), std::move(tau));
}

/*
  Computes the QR decomposition using GEQRF and ORGQR operations.
  This is an in-place function and Q, R tensors must have correct shape and be Fortran contiguous.

  Args:
  * `input` - [in] Input tensor for QR decomposition
  * `Q` - [out] Tensor containing the Q matrices of QR decomposition
  * `R` - [out] Tensor containing the R matrices of QR decomposition
  * `compute_q` - controls whether the Q tensor is computed
  * `reduced_mode` - controls the size of Q and R tensors

  For further details, please see the LAPACK documentation for GEQRF and ORGQR.
*/
TORCH_IMPL_FUNC(linalg_qr_out)(const Tensor& A,
                               std::string_view mode,
                               const Tensor & Q,
                               const Tensor & R) {
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto k = std::min(m, n);
  auto [compute_q, reduced_mode] = at::native::_parse_qr_mode(mode);


  // We need an auxiliary tensor to call geqrf
  auto tau_shape = A.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = k;
  auto tau = A.new_empty(tau_shape);

  // geqrf requires m x n workspace input that is modified in-place
  // We try to use Q. If it doesn't fit, we try to use R
  // If m > n and compute_q==false, it won't fit into Q or R, so we need to create an auxiliary tensor
  Tensor QR;
  if (compute_q && Q.size(-1) == n) {
    QR = Q;
    QR.copy_(A);
  } else if (R.size(-2) == m) {
    QR = R;
    QR.copy_(A);
  } else {
    QR = cloneBatchedColumnMajor(A);
  }

  geqrf_stub(A.device().type(), QR, tau);

  // Split QR into Q (unless compute_q == false) and R
  if (QR.is_alias_of(R)) {
    // Copy QR into Q
    if (compute_q) {
      // If the result didn't fit in Q and compute_q == true is because Q is not of size m x n (i.e. it's of size m x m)
      TORC

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 3 class(es): scalar_t, that, that


## Key Components

The file contains 18436 words across 4100 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 169041 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
