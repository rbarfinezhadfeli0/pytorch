# Documentation: `docs/aten/src/ATen/native/mkldnn/Matmul.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/Matmul.h_docs.md`
- **Size**: 4,729 bytes (4.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/Matmul.h`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/Matmul.h`
- **Size**: 2,394 bytes (2.34 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/native/LinearAlgebraUtils.h>  // For TransposeType

namespace at::native {

// result = beta * result + alpha * gemm(mat1, mat2)
TORCH_API void mkldnn_matmul(
        const Tensor &mat1,
        const Tensor &mat2,
        const Tensor &result,
        float beta=1,
        float alpha=1);

bool use_mkldnn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

bool use_mkldnn_fp16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

bool use_mkldnn_bf32_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

bool use_mkldnn_tf32_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

// Try running mkldnn optimized gemm, or returns false if naive gemm would be faster
bool mkldnn_bf16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a, int64_t lda,
    const c10::BFloat16 *b, int64_t ldb,
    float beta,
    c10::BFloat16 *c, int64_t ldc);

bool mkldnn_bf16f32_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a, int64_t lda,
    const c10::BFloat16 *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc);

bool mkldnn_fp16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::Half *a, int64_t lda,
    const c10::Half *b, int64_t ldb,
    float beta,
    c10::Half *c, int64_t ldc);

/*
oneDNN implicit reduced precision arithmetic feature
https://github.com/mgouicem/oneDNN/tree/mgouicem/rfcs/implicit_downconvert/rfcs/20210301-computation-datatype
to allow implicitly cast data type from FP32 to BF16 in onednn compute primitives
*/
bool mkldnn_reduced_f32_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc);

bool use_mkldnn_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result);

// x:s8 * w:s8 -> y:s32
TORCH_API void mkldnn_matmul_i8i8i32(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result);

}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Config.h`
- `ATen/native/LinearAlgebraUtils.h`


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

Files in the same folder (`aten/src/ATen/native/mkldnn`):

- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`Gelu.cpp_docs.md`](./Gelu.cpp_docs.md)
- [`Conv.h_docs.md`](./Conv.h_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`Matmul.cpp_docs.md`](./Matmul.cpp_docs.md)
- [`TensorShape.cpp_docs.md`](./TensorShape.cpp_docs.md)
- [`RNN.cpp_docs.md`](./RNN.cpp_docs.md)
- [`RegisterMkldnnOpContextClass.cpp_docs.md`](./RegisterMkldnnOpContextClass.cpp_docs.md)
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)


## Cross-References

- **File Documentation**: `Matmul.h_docs.md`
- **Keyword Index**: `Matmul.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/mkldnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/mkldnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/mkldnn`):

- [`ConvPrepack.h_docs.md_docs.md`](./ConvPrepack.h_docs.md_docs.md)
- [`Conv.h_kw.md_docs.md`](./Conv.h_kw.md_docs.md)
- [`IDeepRegistration.h_docs.md_docs.md`](./IDeepRegistration.h_docs.md_docs.md)
- [`Prelu.cpp_kw.md_docs.md`](./Prelu.cpp_kw.md_docs.md)
- [`MKLDNNConversions.cpp_kw.md_docs.md`](./MKLDNNConversions.cpp_kw.md_docs.md)
- [`BinaryOps.cpp_docs.md_docs.md`](./BinaryOps.cpp_docs.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`MkldnnTensorMath.cpp_kw.md_docs.md`](./MkldnnTensorMath.cpp_kw.md_docs.md)
- [`SoftMax.cpp_docs.md_docs.md`](./SoftMax.cpp_docs.md_docs.md)
- [`ConvPrepack.cpp_kw.md_docs.md`](./ConvPrepack.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Matmul.h_docs.md_docs.md`
- **Keyword Index**: `Matmul.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
