# Documentation: `aten/src/ATen/cuda/tunable/GemmRocblas.h`

## File Metadata

- **Path**: `aten/src/ATen/cuda/tunable/GemmRocblas.h`
- **Size**: 10,550 bytes (10.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <c10/util/StringUtil.h>
#include <fmt/printf.h>

#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>

#define TORCH_ROCBLAS_CHECK(EXPR)                 \
  do {                                            \
    rocblas_status __err = EXPR;                  \
    TORCH_CHECK(__err == rocblas_status_success,  \
                "rocblas error: ",                \
                rocblas_status_to_string(__err),  \
                " when calling `" #EXPR "`");     \
  } while (0)

namespace at::cuda::tunable {

template <typename T>
constexpr rocblas_datatype RocBlasDataTypeFor();

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<float>() {
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<double>() {
  return rocblas_datatype_f64_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<Half>() {
  return rocblas_datatype_f16_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<BFloat16>() {
  return rocblas_datatype_bf16_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<c10::complex<float>>() {
  return rocblas_datatype_f32_c;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<c10::complex<double>>() {
  return rocblas_datatype_f64_c;
}

template <typename T>
constexpr rocblas_datatype RocBlasComputeTypeFor();

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<float>() {
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<double>() {
  return rocblas_datatype_f64_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<Half>() {
  // Note that we're returning the _compute_ type for a given datatype.
  // As of 12/2022, using compute type FP16 for 16-bit floats was much
  // slower than using compute type FP32. So we use FP32 compute even for
  // FP16 datatypes. This is how GEMM is implemented even in the function
  // rocblasGemmHelper (see fpgeneric.h)
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<BFloat16>() {
  // Note that we're returning the _compute_ type for a given datatype.
  // As of 12/2022, using compute type FP16 for 16-bit floats was much
  // slower than using compute type FP32. So we use FP32 compute even for
  // BF16 datatypes. This is how GEMM is implemented even in the function
  // rocblasGemmHelper (see fpgeneric.h)
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<c10::complex<float>>() {
  return rocblas_datatype_f32_c;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<c10::complex<double>>() {
  return rocblas_datatype_f64_c;
}

template <typename T>
auto DoCastForHalfOrBfloat16(const T fp) {
  return fp;
}

template <>
inline auto DoCastForHalfOrBfloat16<Half>(const Half fp) {
  // alpha and beta should be the same as compute_type, in Half case it is float.
  float h = fp;
  return h;
}

template <>
inline auto DoCastForHalfOrBfloat16<BFloat16>(const BFloat16 fp) {
  // alpha and beta should be the same as compute_type, in bfloat16 case it is float.
  float h = fp;
  return h;
}

static rocblas_operation _rocblasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return rocblas_operation_none;
    case 't':
    case 'T':
      return rocblas_operation_transpose;
    case 'c':
    case 'C':
      return rocblas_operation_conjugate_transpose;
  }
  TORCH_CHECK(false,
      "_rocblasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

template <typename T>
class RocblasGemmOp : public Callable<GemmParams<T>> {
  public:
    RocblasGemmOp(int solution) : solution_{solution} {}

    TuningStatus Call(const GemmParams<T>* params) override {
      auto input_output_type = RocBlasDataTypeFor<T>();
      if (at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) == at::Float32Precision::TF32 && input_output_type == rocblas_datatype_f32_r)
        return FAIL;  // no support for TF32 in rocBLAS
      auto compute_type = RocBlasComputeTypeFor<T>();
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      auto status = rocblas_gemm_ex(
          (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),
          _rocblasOpFromChar(params->transa),
          _rocblasOpFromChar(params->transb),
          params->m, params->n, params->k,
          &h_a,
          params->a, input_output_type, params->lda,
          params->b, input_output_type, params->ldb,
          &h_b,
          params->c, input_output_type, params->ldc,
          params->c, input_output_type, params->ldc,
          compute_type,
          rocblas_gemm_algo_solution_index,
          solution_,
          rocblas_gemm_flags_none);
      if (status != rocblas_status_success) {
        return FAIL;
      }
      return OK;
    }

  private:
    int solution_;
};

template <typename T>
auto GetRocBlasGemmTypeStringAndOps() {
  rocblas_handle handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();
  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();
  // Get the number of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            nullptr,
                                                            &solution_size));
  std::vector<int> solutions(solution_size);
  // Get the list of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            solutions.data(),
                                                            &solution_size));
  std::vector<std::pair<std::string, std::unique_ptr<Callable<GemmParams<T>>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto callable = std::make_unique<RocblasGemmOp<T>>(solutions[i]);
    ret.emplace_back(std::make_pair(fmt::sprintf("Gemm_Rocblas_%d", solutions[i]), std::move(callable)));
  }
  return ret;
}

template <typename T>
class RocblasGemmStridedBatchedOp : public Callable<GemmStridedBatchedParams<T>> {
  public:
    RocblasGemmStridedBatchedOp(int solution) : solution_{solution} {}

    TuningStatus Call(const GemmStridedBatchedParams<T>* params) override {
      auto input_output_type = RocBlasDataTypeFor<T>();
      if (at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) == at::Float32Precision::TF32 && input_output_type == rocblas_datatype_f32_r)
        return FAIL;  // no support for TF32 in rocBLAS
      auto compute_type = RocBlasComputeTypeFor<T>();
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      auto status = rocblas_gemm_strided_batched_ex(
          (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),
          _rocblasOpFromChar(params->transa),
          _rocblasOpFromChar(params->transb),
          params->m, params->n, params->k,
          &h_a,
          params->a, input_output_type, params->lda, params->stride_a,
          params->b, input_output_type, params->ldb, params->stride_b,
          &h_b,
          params->c, input_output_type, params->ldc, params->stride_c,
          params->c, input_output_type, params->ldc, params->stride_c,
          params->batch,
          compute_type,
          rocblas_gemm_algo_solution_index,
          solution_,
          rocblas_gemm_flags_none);
      if (status != rocblas_status_success) {
        return FAIL;
      }
      return OK;
    }

  private:
    int solution_;
};

template <typename T>
auto GetRocBlasGemmStridedBatchedTypeStringAndOps() {
  rocblas_handle handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();
  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();
  // Get the number of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            nullptr,
                                                            &solution_size));
  std::vector<int> solutions(solution_size);
  // Get the list of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            solutions.data(),
                                                            &solution_size));
  // Sort the solutions in ascending order to make the solution vector deterministic across runs
  std::sort(solutions.begin(), solutions.end());

  std::vector<std::pair<std::string, std::unique_ptr<Callable<GemmStridedBatchedParams<T>>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto callable = std::make_unique<RocblasGemmStridedBatchedOp<T>>(solutions[i]);
    ret.emplace_back(std::make_pair(c10::str("Gemm_Rocblas_", solutions[i]), std::move(callable)));
  }
  return ret;
}

}  // namespace at::cuda::tunable

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `RocblasGemmOp`, `RocblasGemmStridedBatchedOp`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda/tunable`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cuda/CUDAContext.h`
- `ATen/cuda/tunable/TunableOp.h`
- `ATen/cuda/tunable/GemmCommon.h`
- `c10/util/StringUtil.h`
- `fmt/printf.h`
- `rocblas/rocblas.h`


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

Files in the same folder (`aten/src/ATen/cuda/tunable`):

- [`TunableOp.h_docs.md`](./TunableOp.h_docs.md)
- [`GemmHipblaslt.h_docs.md`](./GemmHipblaslt.h_docs.md)
- [`Tunable.h_docs.md`](./Tunable.h_docs.md)
- [`StreamTimer.h_docs.md`](./StreamTimer.h_docs.md)
- [`StreamTimer.cpp_docs.md`](./StreamTimer.cpp_docs.md)
- [`GemmCommon.h_docs.md`](./GemmCommon.h_docs.md)
- [`TunableGemm.h_docs.md`](./TunableGemm.h_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`Tunable.cpp_docs.md`](./Tunable.cpp_docs.md)


## Cross-References

- **File Documentation**: `GemmRocblas.h_docs.md`
- **Keyword Index**: `GemmRocblas.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
