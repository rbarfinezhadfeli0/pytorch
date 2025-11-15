# Documentation: `docs/aten/src/ATen/cuda/tunable/TunableGemm.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/tunable/TunableGemm.h_docs.md`
- **Size**: 12,509 bytes (12.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/tunable/TunableGemm.h`

## File Metadata

- **Path**: `aten/src/ATen/cuda/tunable/TunableGemm.h`
- **Size**: 9,632 bytes (9.41 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <ATen/cuda/tunable/GemmCommon.h>
#ifdef USE_ROCM
#include <ATen/cuda/tunable/GemmHipblaslt.h>
#include <ATen/cuda/tunable/GemmRocblas.h>
#endif
#include <ATen/cuda/tunable/TunableOp.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Float8_e8m0fnu.h>
#include <c10/util/StringUtil.h>
#include <fmt/printf.h>

namespace at::cuda::tunable {

template <typename T>
class DefaultGemmOp : public Callable<GemmParams<T>> {
  public:
    TuningStatus Call(const GemmParams<T>* params) override {
      at::cuda::blas::gemm_internal<T>(
          params->transa, params->transb,
          params->m, params->n, params->k,
          params->alpha,
          params->a, params->lda,
          params->b, params->ldb,
          params->beta,
          params->c, params->ldc);
      return OK;
    }
};

static bool _transposeBoolFromChar(char op) {
  return op == 't' || op == 'T';
}

template <typename T>
class DefaultGemmAndBiasOp : public Callable<GemmAndBiasParams<T>> {
  public:
    TuningStatus Call(const GemmAndBiasParams<T>* params) override {
      at::cuda::blas::gemm_and_bias<T>(
          _transposeBoolFromChar(params->transa),
          _transposeBoolFromChar(params->transb),
          params->m, params->n, params->k,
          params->alpha,
          params->a, params->lda,
          params->b, params->ldb,
          params->bias,
          params->c, params->ldc,
          params->activation);
      return OK;
    }
};

template <typename T>
class DefaultGemmStridedBatchedOp : public Callable<GemmStridedBatchedParams<T>> {
  public:
    TuningStatus Call(const GemmStridedBatchedParams<T>* params) override {
      at::cuda::blas::bgemm_internal<T>(
          params->transa, params->transb,
          params->m, params->n, params->k,
          params->alpha,
          params->a, params->lda, params->stride_a,
          params->b, params->ldb, params->stride_b,
          params->beta,
          params->c, params->ldc, params->stride_c,
          params->batch);
      return OK;
    }
};

template <typename T>
class DefaultScaledGemmOp : public Callable<ScaledGemmParams<T>> {
  public:
    TuningStatus Call(const ScaledGemmParams<T>* params) override {
      at::cuda::blas::scaled_gemm(
          params->transa,
          params->transb,
          params->m,
          params->n,
          params->k,
          params->a,
          params->a_scale_ptr,
          params->lda,
          params->a_dtype,
          params->a_scale_dtype,
          params->a_scaling_type,
          params->b,
          params->b_scale_ptr,
          params->ldb,
          params->b_dtype,
          params->b_scale_dtype,
          params->b_scaling_type,
          params->bias_ptr,
          params->bias_dtype,
          params->c,
          params->c_scale_ptr,
          params->ldc,
          params->c_dtype,
          params->use_fast_accum,
          std::nullopt /* alpha */);
      return OK;
    }
};

template <typename T>
inline bool IsZero(T v) {
  return v == 0.0f;
}

template <>
inline bool IsZero(BFloat16 v) {
  return v.x == 0;
}

template <>
inline bool IsZero(Half v) {
  return float(v) == 0.0f;
}

template <>
inline bool IsZero(c10::complex<double> v) {
  return v == 0.0;
}

template <>
inline bool IsZero(c10::complex<float> v) {
  return v == 0.0f;
}

template <typename T>
inline const char* TypeName(T v) {
  return "unknown";
}

template <>
inline const char* TypeName(float v) {
  if (at::globalContext().allowTF32CuBLAS()) {
    return "tf32";
  } else {
    return "float";
  }
}

template <>
inline const char* TypeName(double v) {
  return "double";
}

template <>
inline const char* TypeName(BFloat16 v) {
  return "BFloat16";
}

template <>
inline const char* TypeName(Half v) {
  return "Half";
}

template <>
inline const char* TypeName(Float8_e4m3fn v) {
  return "Float8_e4m3fn";
}

template <>
inline const char* TypeName(Float8_e5m2 v) {
  return "Float8_e5m2";
}

template <>
inline const char* TypeName(Float8_e4m3fnuz v) {
  return "Float8_e4m3fnuz";
}

template <>
inline const char* TypeName(Float8_e5m2fnuz v) {
  return "Float8_e5m2fnuz";
}

template <>
inline const char* TypeName(Float8_e8m0fnu v) {
  return "Float8_e8m0fnu";
}

template <>
inline const char* TypeName(c10::complex<double> v) {
  return "c10::complex<double>";
}

template <>
inline const char* TypeName(c10::complex<float> v) {
  return "c10::complex<float>";
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
class GemmTunableOp : public TunableOp<GemmParams<T>> {
 public:
  GemmTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmOp<T>>());

#ifdef USE_ROCM
    static const auto env_rocblas = c10::utils::check_env("PYTORCH_TUNABLEOP_ROCBLAS_ENABLED");
    if (!env_rocblas.has_value() || env_rocblas.value()) {
      for (auto&& [name, op] : GetRocBlasGemmTypeStringAndOps<T>()) {
        this->RegisterOp(std::move(name), std::move(op));
      }
    }

    static const auto env_hipblaslt = c10::utils::check_env("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED");
    if (!env_hipblaslt.has_value() || env_hipblaslt.value()) {
      // disallow tuning of hipblaslt with c10::complex
      if constexpr (
          !std::is_same_v<T, c10::complex<float>> &&
          !std::is_same_v<T, c10::complex<double>>) {
        for (auto&& [name, op] : GetHipBlasLtGemmTypeStringAndOps<T, ALayout, BLayout>()) {
          this->RegisterOp(std::move(name), std::move(op));
        }
      }
    }
#endif

    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmOp<T>>());
  }

  std::string Signature() override {
    return fmt::sprintf("GemmTunableOp_%s_%c%c", TypeName<T>(T{}), BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

template <typename T, BlasOp ALayout, BlasOp BLayout>
class GemmAndBiasTunableOp : public TunableOp<GemmAndBiasParams<T>> {
 public:
  GemmAndBiasTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmAndBiasOp<T>>());

#ifdef USE_ROCM
    static const auto env_hipblaslt = c10::utils::check_env("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED");
    if (!env_hipblaslt.has_value() || env_hipblaslt.value()) {
      // disallow tuning of hipblaslt with c10::complex
      if constexpr (
          !std::is_same_v<T, c10::complex<float>> &&
          !std::is_same_v<T, c10::complex<double>>) {
        for (auto&& [name, op] : GetHipBlasLtGemmAndBiasTypeStringAndOps<T, ALayout, BLayout>()) {
          this->RegisterOp(std::move(name), std::move(op));
        }
      }
    }
#endif

    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmAndBiasOp<T>>());
  }

  std::string Signature() override {
    return fmt::sprintf("GemmAndBiasTunableOp_%s_%c%c", TypeName<T>(T{}), BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

template <typename T, BlasOp ALayout, BlasOp BLayout>
class GemmStridedBatchedTunableOp : public TunableOp<GemmStridedBatchedParams<T>> {
 public:
  GemmStridedBatchedTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmStridedBatchedOp<T>>());

#ifdef USE_ROCM
    static const auto env_rocblas = c10::utils::check_env("PYTORCH_TUNABLEOP_ROCBLAS_ENABLED");
    if (!env_rocblas.has_value() || env_rocblas.value()) {
      for (auto&& [name, op] : GetRocBlasGemmStridedBatchedTypeStringAndOps<T>()) {
        this->RegisterOp(std::move(name), std::move(op));
      }
    }

    static const auto env_hipblaslt = c10::utils::check_env("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED");
    if (!env_hipblaslt.has_value() || env_hipblaslt.value()) {
      // disallow tuning of hipblaslt with c10::complex
      if constexpr (
          !std::is_same_v<T, c10::complex<float>> &&
          !std::is_same_v<T, c10::complex<double>>) {
        for (auto&& [name, op] : GetHipBlasLtGemmStridedBatchedTypeStringAndOps<T, ALayout, BLayout>()) {
          this->RegisterOp(std::move(name), std::move(op));
        }
      }
    }
#endif

    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmStridedBatchedOp<T>>());
  }

  std::string Signature() override {
    return fmt::sprintf("GemmStridedBatchedTunableOp_%s_%c%c", TypeName<T>(T{}), BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout>
class ScaledGemmTunableOp : public TunableOp<ScaledGemmParams<CT>> {
 public:
  ScaledGemmTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultScaledGemmOp<CT>>());

#ifdef USE_ROCM
    for (auto&& [name, op] : GetHipBlasLtScaledGemmTypeStringAndOps<AT, BT, CT, ALayout, BLayout>()) {
      this->RegisterOp(std::move(name), std::move(op));
    }
#endif

    this->RegisterOp(std::string("Default"), std::make_unique<DefaultScaledGemmOp<CT>>());
  }

  std::string Signature() override {
    return fmt::sprintf("ScaledGemmTunableOp_%s_%s_%s_%c%c",
      TypeName<AT>(AT{}),
      TypeName<BT>(BT{}),
      TypeName<CT>(CT{}),
      BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

} // namespace at::cuda::tunable

```



## High-Level Overview


This C++ file contains approximately 8 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `DefaultGemmOp`, `DefaultGemmAndBiasOp`, `DefaultGemmStridedBatchedOp`, `DefaultScaledGemmOp`, `GemmTunableOp`, `GemmAndBiasTunableOp`, `GemmStridedBatchedTunableOp`, `ScaledGemmTunableOp`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda/tunable`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cuda/tunable/GemmCommon.h`
- `ATen/cuda/tunable/GemmHipblaslt.h`
- `ATen/cuda/tunable/GemmRocblas.h`
- `ATen/cuda/tunable/TunableOp.h`
- `c10/cuda/CUDACachingAllocator.h`
- `c10/util/Float8_e4m3fn.h`
- `c10/util/Float8_e4m3fnuz.h`
- `c10/util/Float8_e5m2.h`
- `c10/util/Float8_e5m2fnuz.h`
- `c10/util/Float8_e8m0fnu.h`
- `c10/util/StringUtil.h`
- `fmt/printf.h`


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
- [`README.md_docs.md`](./README.md_docs.md)
- [`Tunable.cpp_docs.md`](./Tunable.cpp_docs.md)
- [`GemmRocblas.h_docs.md`](./GemmRocblas.h_docs.md)


## Cross-References

- **File Documentation**: `TunableGemm.h_docs.md`
- **Keyword Index**: `TunableGemm.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cuda/tunable`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cuda/tunable`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/cuda/tunable`):

- [`Tunable.cpp_docs.md_docs.md`](./Tunable.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`TunableOp.h_kw.md_docs.md`](./TunableOp.h_kw.md_docs.md)
- [`StreamTimer.cpp_kw.md_docs.md`](./StreamTimer.cpp_kw.md_docs.md)
- [`GemmRocblas.h_kw.md_docs.md`](./GemmRocblas.h_kw.md_docs.md)
- [`TunableGemm.h_kw.md_docs.md`](./TunableGemm.h_kw.md_docs.md)
- [`GemmHipblaslt.h_docs.md_docs.md`](./GemmHipblaslt.h_docs.md_docs.md)
- [`GemmCommon.h_kw.md_docs.md`](./GemmCommon.h_kw.md_docs.md)
- [`Tunable.h_kw.md_docs.md`](./Tunable.h_kw.md_docs.md)
- [`GemmHipblaslt.h_kw.md_docs.md`](./GemmHipblaslt.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `TunableGemm.h_docs.md_docs.md`
- **Keyword Index**: `TunableGemm.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
