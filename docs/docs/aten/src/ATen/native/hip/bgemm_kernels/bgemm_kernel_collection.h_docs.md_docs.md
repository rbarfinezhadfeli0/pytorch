# Documentation: `docs/aten/src/ATen/native/hip/bgemm_kernels/bgemm_kernel_collection.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/hip/bgemm_kernels/bgemm_kernel_collection.h_docs.md`
- **Size**: 5,505 bytes (5.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/hip/bgemm_kernels/bgemm_kernel_collection.h`

## File Metadata

- **Path**: `aten/src/ATen/native/hip/bgemm_kernels/bgemm_kernel_collection.h`
- **Size**: 3,454 bytes (3.37 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>

namespace at::native {
void bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v4(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_16x16x1_16x16x1_1x16x1x16_4_Intrawave_v4(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_4x64x1_4x64x1_1x16x1x16_4_Intrawave_v3(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_4x64x1_4x64x1_1x16x1x16_4_Intrawave_v5(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v5(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v1(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_32x16x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2_Intrawave_v1(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_64_16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_Intrawave_v1(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_16x32x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v1(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_16x32x64_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4_Intrawave_v1(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_16x32x64_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4_Intrawave_v1(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_256x16x64_16x16_4x1_8x32x1_8x16x1_1x32x1x8_2_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_256x16x64_16x16_4x1_16x16x1_16x8x1_1x32x1x8_2_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_256x16x64_16x16_4x1_32x8x1_32x4x1_1x32x1x8_2_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_128x16x64_16x16_4x1_8x16x1_8x16x1_1x16x1x8_2_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_64x16x64_16x16_2x1_8x16x1_8x16x1_1x16x1x8_2_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_32x16x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_64_16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_16x32x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_128_16x128x64_16x16_1x4_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
void bgemm_kernel_bf16bf16bf16_256_16x256x64_16x16_1x4_8x16x1_8x16x1_1x16x1x16_4_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));

}; // namespace at::native
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 25 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/hip/bgemm_kernels`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/OpMathType.h`
- `ATen/hip/HIPBlas.h`


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

Files in the same folder (`aten/src/ATen/native/hip/bgemm_kernels`):

- [`bgemm_kernel_template.h_docs.md`](./bgemm_kernel_template.h_docs.md)


## Cross-References

- **File Documentation**: `bgemm_kernel_collection.h_docs.md`
- **Keyword Index**: `bgemm_kernel_collection.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/hip/bgemm_kernels`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/hip/bgemm_kernels`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/hip/bgemm_kernels`):

- [`bgemm_kernel_collection.h_kw.md_docs.md`](./bgemm_kernel_collection.h_kw.md_docs.md)
- [`bgemm_kernel_template.h_kw.md_docs.md`](./bgemm_kernel_template.h_kw.md_docs.md)
- [`bgemm_kernel_template.h_docs.md_docs.md`](./bgemm_kernel_template.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `bgemm_kernel_collection.h_docs.md_docs.md`
- **Keyword Index**: `bgemm_kernel_collection.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
