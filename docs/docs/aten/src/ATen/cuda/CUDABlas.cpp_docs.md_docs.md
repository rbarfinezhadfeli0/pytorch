# Documentation: `docs/aten/src/ATen/cuda/CUDABlas.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/CUDABlas.cpp_docs.md`
- **Size**: 53,130 bytes (51.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/CUDABlas.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cuda/CUDABlas.cpp`
- **Size**: 96,978 bytes (94.71 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
/*
  Provides the implementations of CUDA BLAS function templates.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Export.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <c10/core/ScalarType.h>

#include <ATen/cuda/detail/BLASConstants.h>

#ifdef USE_ROCM
#include <c10/cuda/CUDAStream.h>
#include <hipblaslt/hipblaslt-ext.hpp>
// until hipblas has an API to accept flags, we must use rocblas here
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <ATen/native/hip/ck_gemm.h>
#include <ATen/native/hip/ck_bgemm.h>
#define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 + ROCBLAS_VERSION_MINOR)
#define USE_GEMM_FLAGS_FP16_ALT_IMPL (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242)
// needed to work around calling rocblas API instead of hipblas API
static rocblas_operation hipOperationToRocOperation(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return rocblas_operation_none;
    case HIPBLAS_OP_T:
        return rocblas_operation_transpose;
    case HIPBLAS_OP_C:
        return rocblas_operation_conjugate_transpose;
    }
    TORCH_CHECK(false, "HIPBLAS_STATUS_INVALID_ENUM");
}
static hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    TORCH_CHECK(false, "HIPBLAS_STATUS_INVALID_ENUM");
}
// hipblas does not have hipblasSetMathMode
#define hipblasSetMathMode(handle, flags) HIPBLAS_STATUS_SUCCESS
// until we use hiblas v2
// hipify correctly maps things like CUDA_R_16F to HIP_R_16F,
// however hipblas v1 is still using its custom type
#ifndef HIPBLAS_V2
#define HIP_R_16F  HIPBLAS_R_16F
#define HIP_R_32F  HIPBLAS_R_32F
#define HIP_R_64F  HIPBLAS_R_64F
#define HIP_C_16F  HIPBLAS_C_16F
#define HIP_C_32F  HIPBLAS_C_32F
#define HIP_C_64F  HIPBLAS_C_64F
#define HIP_R_8I   HIPBLAS_R_8I
#define HIP_R_8U   HIPBLAS_R_8U
#define HIP_R_32I  HIPBLAS_R_32I
#define HIP_R_32U  HIPBLAS_R_32U
#define HIP_C_8I   HIPBLAS_C_8I
#define HIP_C_8U   HIPBLAS_C_8U
#define HIP_C_32I  HIPBLAS_C_32I
#define HIP_C_32U  HIPBLAS_C_32U
#define HIP_R_16BF HIPBLAS_R_16B
#define HIP_C_16BF HIPBLAS_C_16B
#endif
#endif

#define CUDABLAS_POSINT_CHECK(FD, X)         \
  TORCH_CHECK(                               \
      (X > 0 && X <= INT_MAX),               \
      "at::cuda::blas::" #FD " argument " #X \
      " must be positive and less than ",    \
      INT_MAX,                               \
      " but got ",                           \
      X)

#define CUDABLAS_NONNEGINT_CHECK(FD, X)       \
  TORCH_CHECK(                                \
      (X >= 0 && X <= INT_MAX),               \
      "at::cuda::blas::" #FD " argument " #X  \
      " must be non-negative and less than ", \
      INT_MAX,                                \
      " but got ",                            \
      X)

namespace {

cublasOperation_t _cublasOpFromChar(char op) {
  // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
  switch (op) {
    case 'n':
      [[fallthrough]];
    case 'N':
      return CUBLAS_OP_N;
    case 't':
      [[fallthrough]];
    case 'T':
      return CUBLAS_OP_T;
    case 'c':
      [[fallthrough]];
    case 'C':
      return CUBLAS_OP_C;
  }
  TORCH_CHECK(false,
      "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

void _cublasAdjustLdLevel2(int64_t m, int64_t n, int64_t* lda) {
  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).

  // Q: Why does Level3 check trans but this doesn't?
  // A: In level 2, the sizes (m, n) specify the size of A
  // (independent of trans value). In level 3. the sizes (m, n, k)
  // specify the sizes of op(A), op(B) where op depend on trans
  // values.
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);
}

void _cublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}

#ifndef USE_ROCM
uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (; ; alignment /= 2) {
    if (!(address % alignment)) {
      return alignment;
    }
  }
}
#endif

#ifdef USE_ROCM
static c10::cuda::CUDAStream _getCarveoutStream(int32_t value) {
  // 0 is default value, meaning full CUs i.e. no mask
  if (value == 0) {
    return at::cuda::getCurrentCUDAStream();
  }
  static int32_t last_value = 0;
  static hipStream_t stream;
  if (last_value == 0) {
    // first request, do nothing for this case
  }
  else if (last_value == value) {
    // stream was created previously and value hasn't changed
    return c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());
  }
  else {
    // need a new stream and a previous stream exists, delete it
    AT_CUDA_CHECK(hipStreamDestroy(stream));
  }

  // if we got here, we need to create a new stream
  int32_t CUs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // how many uint32_t do we need to cover all CUs, fill bitmask with 1
  uint32_t mask_size = static_cast<uint32_t>((CUs + 32 - 1) / 32);
  std::vector<uint32_t> mask(mask_size, uint32_t{0x00000000});
  // starting from lowest order bits, in 32-bit chunks
  // set bits to 0 based on how many CUs to carve out
  int32_t full_shifts = value / 32;
  int32_t remainder = value % 32;
  for (int32_t i = 0; i < full_shifts; i++) {
    mask[i] = uint32_t{0xffffffff};
  }
  mask[full_shifts] = uint32_t{0xffffffff} << (32 - remainder);

  // finally, create masked stream
  AT_CUDA_CHECK(hipExtStreamCreateWithCUMask(&stream, mask_size, &mask[0]));

  last_value = value;
  return c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());
}

static void _syncCurrentWithCarveoutStream(hipStream_t stream, bool presync) {
  hipEvent_t event;
  AT_CUDA_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));

  auto current_stream = at::cuda::getCurrentCUDAStream();

  if (presync) {
    AT_CUDA_CHECK(hipEventRecord(event, current_stream));
    AT_CUDA_CHECK(hipStreamWaitEvent(stream, event, 0));
  }
  else {
    AT_CUDA_CHECK(hipEventRecord(event, stream));
    AT_CUDA_CHECK(hipStreamWaitEvent(current_stream, event, 0));
  }
}
#endif

struct CublasLtWorkspace {
  CublasLtWorkspace() {
    size = at::cuda::getCUDABlasLtWorkspaceSize();
    ptr = at::cuda::getCUDABlasLtWorkspace();
  }
  void * ptr;
  size_t size;
};
} // anonymous namespace

namespace at::cuda::blas {

/* LEVEL 3 BLAS FUNCTIONS */

#define GEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldc);  \
  } while (0)

#define BGEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldc);  \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, num_batches);  \
  } while (0)

namespace {
// Following the pattern of CuSparseDescriptor
// Defined here for now because this is the only place cublas_lt interface is
// used but can be moved to a header once cublas_lt interface is used in
// multiple places.
template <typename T, cublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDABLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, cublasStatus_t (*destructor)(T*)>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};

class CuBlasLtMatmulDescriptor : public CuBlasLtDescriptor<
                                     cublasLtMatmulDescOpaque_t,
                                     &cublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(
      cublasComputeType_t compute_type,
      cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  void setAttribute(cublasLtMatmulDescAttributes_t attr, const T value) {
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(value)));
  }
};

class CuBlasLtMatrixLayout : public CuBlasLtDescriptor<
                                 cublasLtMatrixLayoutOpaque_t,
                                 &cublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtMatrixLayout(
      cudaDataType_t type,
      uint64_t rows,
      uint64_t cols,
      int64_t ld,
      bool t = false) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatrixLayoutCreate(&raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  void setAttribute(cublasLtMatrixLayoutAttribute_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatrixLayoutSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<
                                     cublasLtMatmulPreferenceOpaque_t,
                                     &cublasLtMatmulPreferenceDestroy> {
 public:
  CuBlasLtMatmulPreference() {
    cublasLtMatmulPreference_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  void setAttribute(cublasLtMatmulPreferenceAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulPreferenceSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};
} // namespace


template <typename Dtype, typename C_Dtype = Dtype>
static inline bool bgemm_internal_cublaslt(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
#if defined(USE_ROCM) && ROCM_VERSION == 60400
  // regression in ROCm 6.4, planned fixed in 6.4.1, hipblaslt TT fp32 calculation errors
  // best to disallow hipblaslt for this specific case
  if constexpr (std::is_same_v<Dtype, float>) {
    if (_cublasOpFromChar(transa) == CUBLAS_OP_T && _cublasOpFromChar(transb) == CUBLAS_OP_T) {
        return false;
    }
  }
#endif
  cudaDataType_t abType = CUDA_R_32F;
  cudaDataType_t cType = CUDA_R_32F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  CuBlasLtMatmulPreference preference;
#ifndef USE_ROCM
  at::Half halpha;
  at::Half hbeta;
  uint32_t mask = -1;
#endif
  void * alpha_ptr = &alpha;
  void * beta_ptr = &beta;
  if constexpr (std::is_same_v<Dtype, double>) {
    abType = CUDA_R_64F;
    cType = CUDA_R_64F;
    computeType = CUBLAS_COMPUTE_64F;
    scaleType = CUDA_R_64F;
  } else if constexpr (std::is_same_v<Dtype, float>) {
    if (at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) == at::Float32Precision::TF32) {
      computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
  } else if constexpr (std::is_same_v<Dtype, c10::complex<double>>) {
    abType = CUDA_C_64F;
    cType = CUDA_C_64F;
    computeType = CUBLAS_COMPUTE_64F;
    scaleType = CUDA_C_64F;
  } else if constexpr (std::is_same_v<Dtype, c10::complex<float>>) {
    abType = CUDA_C_32F;
    cType = CUDA_C_32F;
    scaleType = CUDA_C_32F;
  } else if constexpr (std::is_same_v<Dtype, at::Half>) {
#ifndef USE_ROCM
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major >= 7 && at::globalContext().allowFP16AccumulationCuBLAS()) {
      computeType = CUBLAS_COMPUTE_16F;
      scaleType = CUDA_R_16F;
      halpha = alpha;
      hbeta = beta;
      alpha_ptr = &halpha;
      beta_ptr = &hbeta;
    }
#endif
    abType = CUDA_R_16F;
    cType = (std::is_same_v<C_Dtype, float>) ? CUDA_R_32F : CUDA_R_16F;
#ifndef USE_ROCM
    auto fp16_reduction = at::globalContext().allowFP16ReductionCuBLAS();
    if (fp16_reduction !=
        at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
      mask =
          fp16_reduction ==
                  at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
              ? (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE |
                 CUBLASLT_REDUCTION_SCHEME_NONE)
              : CUBLASLT_REDUCTION_SCHEME_NONE;
      preference.setAttribute(
          CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask);
    }
#endif
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    abType = CUDA_R_16BF;
    cType = (std::is_same_v<C_Dtype, float>) ? CUDA_R_32F : CUDA_R_16BF;
#ifndef USE_ROCM
    auto bf16_reduction = at::globalContext().allowBF16ReductionCuBLAS();
    if (bf16_reduction !=
        at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
      mask =
          bf16_reduction ==
                  at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
              ? (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE |
                 CUBLASLT_REDUCTION_SCHEME_NONE)
              : CUBLASLT_REDUCTION_SCHEME_NONE;
      preference.setAttribute(
          CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask);
    }
#endif
  } else {
    static_assert(false && sizeof(Dtype), "at::cuda::blas::bgemm_internal_cublaslt: not implemented");
  }

  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb);
  auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    computeDesc.setAttribute<int32_t>(
        CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
            at::globalContext()._SMCarveout_EXPERIMENTAL().value());
  }
#else
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    stream = _getCarveoutStream(
        at::globalContext()._SMCarveout_EXPERIMENTAL().value());
    _syncCurrentWithCarveoutStream(stream, true);
  }
#endif
  CuBlasLtMatrixLayout Adesc(abType, m, k, lda, opa == CUBLAS_OP_T);
  CuBlasLtMatrixLayout Bdesc(abType, k, n, ldb, opb == CUBLAS_OP_T);
  CuBlasLtMatrixLayout Cdesc(cType, m, n, ldc);

  if (num_batches > 1) {
    int num_batches_as_int = static_cast<int>(num_batches);
    Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridea);
    Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideb);
    Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridec);
  }

#ifndef USE_ROCM
  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(a));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(b));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(c));
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
#endif

  auto ltworkspace = CublasLtWorkspace();
  TORCH_CHECK(ltworkspace.ptr != nullptr, "OOM trying to allocate workspace for cublaslt");
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ltworkspace.size);

  cublasStatus_t cublasStatus = CUBLAS_STATUS_SUCCESS;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  // on Blackwell+, we fake a n > 1 matmul when querying heuristics
  // to prevent cuBLASLt from dispatching to a GEMV kernel for batch-invariance
#ifndef USE_ROCM
  const bool lie_to_cublaslt = mask == CUBLASLT_REDUCTION_SCHEME_NONE && n == 1 && at::cuda::getCurrentDeviceProperties()->major >= 10;
#else
  const bool lie_to_cublaslt = false;
#endif
  if (lie_to_cublaslt) {
     CuBlasLtMatrixLayout FakeBdesc(abType, k, 2, ldb, opb == CUBLAS_OP_T);
     CuBlasLtMatrixLayout FakeCdesc(cType, m, 2, ldc);

     TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        computeDesc.descriptor(),
        Adesc.descriptor(),
        FakeBdesc.descriptor(),
        FakeCdesc.descriptor(),
        FakeCdesc.descriptor(),
        preference.descriptor(),
        1,
        &heuristicResult,
        &returnedResult));
  } else {
    TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        computeDesc.descriptor(),
        Adesc.descriptor(),
        Bdesc.descriptor(),
        Cdesc.descriptor(),
        Cdesc.descriptor(),
        preference.descriptor(),
        1,
        &heuristicResult,
        &returnedResult));
  }
  if (returnedResult == 0) {
    cublasStatus = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  else {
    cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      alpha_ptr,
      a,
      Adesc.descriptor(),
      b,
      Bdesc.descriptor(),
      beta_ptr,
      c,
      Cdesc.descriptor(),
      c,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      ltworkspace.ptr,
      ltworkspace.size,
      stream);
#ifdef USE_ROCM
    if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
      _syncCurrentWithCarveoutStream(stream, false);
    }
#endif
  }
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    TORCH_WARN(
      "bgemm_internal_cublaslt error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      (opa == CUBLAS_OP_T),
      " transpose_mat2 ",
      (opb == CUBLAS_OP_T),
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " lda ",
      lda,
      " ldb ",
      ldb,
      " ldc ",
      ldc,
      " abType ",
      abType,
      " cType ",
      cType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType,
      ". Will attempt to recover by calling cublas instead.");
    return false;
  }
  return true;
}


template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm_internal_cublas(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::bgemm: not implemented for input type ", typeid(Dtype).name(), " and output type ", typeid(C_Dtype).name());
}

template <>
void bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(cublasDgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, stridea, reinterpret_cast<const cuDoubleComplex*>(b), ldb, strideb, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_CUDABLAS_CHECK(cublasCgemmStridedBatched(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, stridea, reinterpret_cast<const cuComplex*>(b), ldb, strideb, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc, stridec, num_batches));
}

template <typename C_Dtype>
inline void bgemm_internal_cublas_half_helper(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, C_Dtype)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(at::Half);
  float falpha = alpha;
  float fbeta = beta;
#ifndef USE_ROCM
  at::Half halpha;
  at::Half hbeta;
  auto compute_type = CUDA_R_32F;
#endif
  void * alpha_ptr = &falpha;
  void * beta_ptr = &fbeta;
#ifdef USE_ROCM
  int flag = 0;
  rocblas_datatype c_type = std::is_same<C_Dtype, float>::value ? rocblas_datatype_f32_r : rocblas_datatype_f16_r;
  rocblas_datatype d_type = c_type;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                   hipOperationToRocOperation(opa),
                                   hipOperationToRocOperation(opb), (int)m, (int)n, (int)k,
                                   (void*)alpha_ptr, a, rocblas_datatype_f16_r, (int)lda, stridea,
                                   b, rocblas_datatype_f16_r, (int)ldb, strideb,
                                   (void*)beta_ptr, c, c_type, (int)ldc, stridec,
                                   c, d_type, (int)ldc, stridec,
                                   (int) num_batches, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                   0, flag)));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 7 && at::globalContext().allowFP16AccumulationCuBLAS()) {
    halpha = alpha;
    hbeta = beta;
    compute_type = CUDA_R_16F;
    alpha_ptr = &halpha;
    beta_ptr = &hbeta;
  }
  if (prop->major >= 5){
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, opa, opb, m, n, k,
      alpha_ptr, a, CUDA_R_16F, lda, stridea,
      b, CUDA_R_16F, ldb, strideb, beta_ptr,
      c, std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16F, ldc, stridec,
      num_batches, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    for (const auto i : c10::irange(num_batches)) {
      if (std::is_same_v<C_Dtype, float>) {
        float* c_ptr = (float*)(c + i * stridec);
        at::cuda::blas::gemm<at::Half, float>(
            transa, transb,
            m, n, k,
            alpha, (a + i * stridea), lda,
            (b + i * strideb), ldb, beta,
            c_ptr, ldc);
      } else {
        at::cuda::blas::gemm<at::Half>(
            transa, transb,
            m, n, k,
            alpha, (a + i * stridea), lda,
            (b + i * strideb), ldb, beta,
            (c + i * stridec), ldc);
      }
    }
  }
#endif // USE_ROCM
}

template <typename C_Dtype>
inline void bgemm_internal_cublas_bfloat16_helper(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, C_Dtype)) {
  BGEMM_CHECK_ARGVALUES(at::BFloat16);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  const float falpha = alpha;
  const float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

#if defined(USE_ROCM)
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                              opa, opb, (int)m, (int)n, (int)k,
                              (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea,
                              b, CUDA_R_16BF, (int)ldb, strideb,
                              (void*)&fbeta, c, std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16BF,
                              (int)ldc, stridec, (int)num_batches,
                              compute_type,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <>
void bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  bgemm_internal_cublas_half_helper<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
}

template <>
void bgemm_internal_cublas<at::Half, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  bgemm_internal_cublas_half_helper<float>(CUDABLAS_BGEMM_ARGS(at::Half));
}

template <>
void bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  bgemm_internal_cublas_bfloat16_helper<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}


template <>
void bgemm_internal_cublas<at::BFloat16, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  bgemm_internal_cublas_bfloat16_helper<float>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}


template <>
void bgemm_internal<double>(CUDABLAS_BGEMM_ARGTYPES(double))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support double gemm yet
    bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGS(double));
#else
    if (!bgemm_internal_cublaslt<double>(CUDABLAS_BGEMM_ARGS(double))) {
      bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGS(double));
    }
#endif
  }
  else {
    bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGS(double));
  }
}

template <>
void bgemm_internal<float>(CUDABLAS_BGEMM_ARGTYPES(float))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<float>(CUDABLAS_BGEMM_ARGS(float))) {
      bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGS(float));
    }
  }
  else {
    bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGS(float));
  }
}

template <>
void bgemm_internal<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support complex<double> gemm yet
    bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
#else
    if (!bgemm_internal_cublaslt<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>))) {
      bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
    }
#endif
  }
  else {
    bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
}

template <>
void bgemm_internal<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support complex<float> gemm yet
    bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
#else
    if (!bgemm_internal_cublaslt<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>))) {
      bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
    }
#endif
  }
  else {
    bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
}

template <>
void bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half))) {
      bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
    }
  }
  else {
    bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16))) {
      bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    }
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    at::native::bgemm_internal_ck<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
#endif
  else {
    bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}

template<>
void bgemm_internal<at::Half, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float))
{
  if (at::globalContext().allowFP16AccumulationCuBLAS()) {
    // Do not allow fp16 reductions with fp32 output
    TORCH_CHECK(false, "bgemm input type at::Half and output type float is not supported with allowFP16AccumulationCuBLAS");
  }

  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<at::Half, float>(CUDABLAS_BGEMM_ARGS(at::Half))) {
      bgemm_internal_cublas<at::Half, float>(CUDABLAS_BGEMM_ARGS(at::Half));
    }
  }
#if defined(USE_ROCM) && !defined(_MSC_VER)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    TORCH_CHECK(false, "gemm input type at::Half and output type float is not supported for ROCm");
  }
#endif
  else {
    bgemm_internal_cublas<at::Half, float>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template<>
void bgemm_internal<at::BFloat16, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<at::BFloat16, float>(CUDABLAS_BGEMM_ARGS(at::BFloat16))) {
      bgemm_internal_cublas<at::BFloat16, float>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    }
  }
#if defined(USE_ROCM) && !defined(_MSC_VER)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    TORCH_CHECK(false, "gemm input type at::BFloat16 and output type float is not supported for ROCm");
  }
#endif
  else {
    bgemm_internal_cublas<at::BFloat16, float>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}

template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm_tunable(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  tunable::GemmStridedBatchedParams<Dtype> params;
  params.transa = transa;
  params.transb = transb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = alpha;
  params.a = a;
  params.lda = lda;
  params.stride_a = stridea;
  params.b = b;
  params.ldb = ldb;
  params.stride_b = strideb;
  params.beta = beta;
  params.c = c;
  params.ldc = ldc;
  params.stride_c = stridec;
  params.batch = num_batches;

  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  if (transa_ && transb_) {
    static tunable::GemmStridedBatchedTunableOp<Dtype, tunable::BlasOp::T, tunable::BlasOp::T> bgemm{};
    bgemm(&params);
  }
  else if (transa_ && !transb_) {
    static tunable::GemmStridedBatchedTunableOp<Dtype, tunable::BlasOp::T, tunable::BlasOp::N> bgemm{};
    bgemm(&params);
  }
  else if (!transa_ && transb_) {
    static tunable::GemmStridedBatchedTunableOp<Dtype, tunable::BlasOp::N, tunable::BlasOp::T> bgemm{};
    bgemm(&params);
  }
  else if (!transa_ && !transb_) {
    static tunable::GemmStridedBatchedTunableOp<Dtype, tunable::BlasOp::N, tunable::BlasOp::N> bgemm{};
    bgemm(&params);
  }
  else {
    TORCH_CHECK(false, "unreachable");
  }
}

template <>
void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<double>(CUDABLAS_BGEMM_ARGS(double));
  }
  else {
    bgemm_internal<double>(CUDABLAS_BGEMM_ARGS(double));
  }
}

template <>
void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<float>(CUDABLAS_BGEMM_ARGS(float));
  }
  else {
    bgemm_internal<float>(CUDABLAS_BGEMM_ARGS(float));
  }
}

template <>
void bgemm<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
  else {
    bgemm_internal<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
}

template <>
void bgemm<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
  else {
    bgemm_internal<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
}

template <>
void bgemm<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
  else {
    bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
  else {
    bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}

template <>
void bgemm<at::Half, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  // TODO: Support tuning for Half inputs and FP32 output
  bgemm_internal<at::Half, float>(CUDABLAS_BGEMM_ARGS(at::Half));
}


template <>
void bgemm<at::BFloat16, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  #ifndef USE_ROCM
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();

    if (prop->major < 8)
      TORCH_CHECK(false, "bgemm input type at::BFloat16 and output type float is only supported for CUDA devices with compute capability 8.0 or higher");
  #endif
  // TODO: Support tuning for BFloat16 inputs and FP32 output
  bgemm_internal<at::BFloat16, float>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}



template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm_internal_cublas(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::gemm: not implemented for input type ", typeid(Dtype).name(), " and output type ", typeid(C_Dtype).name());
}

template <>
void gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(cublasDgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm_internal_cublas<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_CUDABLAS_CHECK(cublasZgemm(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc));
}

template <>
void gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_CUDABLAS_CHECK(cublasCgemm(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, reinterpret_cast<const cuComplex*>(b), ldb, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc));
}

template <typename C_Dtype>
inline void gemm_internal_cublas_half_helper(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, C_Dtype)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
#ifndef USE_ROCM
  at::Half halpha;
  at::Half hbeta;
  auto compute_type = CUDA_R_32F;
#endif
  void * alpha_ptr = &falpha;
  void * beta_ptr = &fbeta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::Half);
#ifdef USE_ROCM
  int flag = 0;
  rocblas_datatype c_type = std::is_same<C_Dtype, float>::value ? rocblas_datatype_f32_r : rocblas_datatype_f16_r;
  rocblas_datatype d_type = c_type;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_ex(
      (rocblas_handle)handle,
      hipOperationToRocOperation(opa),
      hipOperationToRocOperation(opb),
      m,
      n,
      k,
      alpha_ptr,
      a,
      rocblas_datatype_f16_r,
      lda,
      b,
      rocblas_datatype_f16_r,
      ldb,
      beta_ptr,
      c,
      c_type,
      ldc,
      c,
      d_type,
      ldc,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      flag)));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 7 && at::globalContext().allowFP16AccumulationCuBLAS()) {
    compute_type = CUDA_R_16F;
    halpha = alpha;
    hbeta = beta;
    alpha_ptr = &halpha;
    beta_ptr = &hbeta;
  }
  if (prop->major >= 5) {
    cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
    auto fp16_reduction = at::globalContext().allowFP16ReductionCuBLAS();
    TORCH_CHECK(fp16_reduction !=
        at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
          "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction("
          "..., allow_splitk=False) requires the cuBLASLt backend");
    if (fp16_reduction !=
        at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
      cublas_flags = static_cast<cublasMath_t>(
          cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    }
    // Disallow fp16 reductions that could lead to unexpected overflow issues.
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        alpha_ptr,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        beta_ptr,
        c,
        std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16F,
        ldc,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSgemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16F,
        ldc));
  }
#endif
}

template <typename C_Dtype>
inline void gemm_internal_cublas_bfloat16_helper(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, C_Dtype)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::BFloat16);
#ifndef USE_ROCM
  cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
  auto bf16_reduction = at::globalContext().allowBF16ReductionCuBLAS();
  TORCH_CHECK(bf16_reduction !=
      at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction("
        "..., allow_splitk=False) requires the cuBLASLt backend");
  if (bf16_reduction !=
      at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
    cublas_flags = static_cast<cublasMath_t>(
        cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  }
#endif
#if defined(USE_ROCM)
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &falpha,
      a,
      CUDA_R_16BF,
      lda,
      b,
      CUDA_R_16BF,
      ldb,
      &fbeta,
      c,
      std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16BF,
      ldc,
      compute_type,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

template <>
void gemm_internal_cublas<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  gemm_internal_cublas_half_helper<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
}

template <>
void gemm_internal_cublas<at::Half, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  gemm_internal_cublas_half_helper<float>(CUDABLAS_GEMM_ARGS(at::Half));
}

template <>
void gemm_internal_cublas<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  gemm_internal_cublas_bfloat16_helper<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
}

template <>
void gemm_internal_cublas<at::BFloat16, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  gemm_internal_cublas_bfloat16_helper<float>(CUDABLAS_GEMM_ARGS(at::BFloat16));
}

template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm_internal_cublaslt(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  // forward to bgemm implementation but set strides and batches to 0
  if (!bgemm_internal_cublaslt(transa, transb, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 0)) {
    gemm_internal_cublas(CUDABLAS_GEMM_ARGS(Dtype));
  }
}

template <>
void gemm_internal<double>(CUDABLAS_GEMM_ARGTYPES(double))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support double gemm yet
    gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGS(double));
#else
    gemm_internal_cublaslt<double>(CUDABLAS_GEMM_ARGS(double));
#endif
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    at::native::gemm_internal_ck<double>(CUDABLAS_GEMM_ARGS(double));
  }
#endif
  else {
    gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGS(double));
  }
}

template <>
void gemm_internal<float>(CUDABLAS_GEMM_ARGTYPES(float))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<float>(CUDABLAS_GEMM_ARGS(float));
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    if (at::detail::getCUDAHooks().isGPUArch({"gfx11", "gfx12"})) { //no CK GEMM version
      gemm_internal_cublaslt<float>(CUDABLAS_GEMM_ARGS(float));
    } else{
      at::native::gemm_internal_ck<float>(CUDABLAS_GEMM_ARGS(float));
    }
  }
#endif
  else {
    gemm_internal_cublas<float>(CUDABLAS_GEMM_ARGS(float));
  }
}

template <>
void gemm_internal<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support complex gemm yet
    gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
#else
    gemm_internal_cublaslt<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
#endif
  }
  else {
    gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
  }
}

template <>
void gemm_internal<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support complex gemm yet
    gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
#else
    gemm_internal_cublaslt<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
#endif
  }
  else {
    gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
  }
}

template <>
void gemm_internal<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    at::native::gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
#endif
  else {
    gemm_internal_cublas<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
}

template <>
void gemm_internal<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    at::native::gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
#endif
  else {
    gemm_internal_cublas<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
}

template<>
void gemm_internal<at::Half, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float))
{
  if (at::globalContext().allowFP16AccumulationCuBLAS()) {
    // Do not allow fp16 reductions with fp32 output
    TORCH_CHECK(false, "gemm input type at::Half and output type float is not supported with allowFP16AccumulationCuBLAS");
  }

  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    gemm_i
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

- **File Documentation**: `CUDABlas.cpp_docs.md_docs.md`
- **Keyword Index**: `CUDABlas.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
