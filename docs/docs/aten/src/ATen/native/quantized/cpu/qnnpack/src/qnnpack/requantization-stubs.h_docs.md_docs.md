# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/requantization-stubs.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/requantization-stubs.h_docs.md`
- **Size**: 5,490 bytes (5.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/requantization-stubs.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/requantization-stubs.h`
- **Size**: 2,989 bytes (2.92 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/params.h>

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*pytorch_requantization_function)(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output);

#define DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(fn_name) \
  void fn_name(                                  \
      size_t n,                                  \
      const int32_t* input,                      \
      float scale,                               \
      uint8_t zero_point,                        \
      uint8_t qmin,                              \
      uint8_t qmax,                              \
      uint8_t* output);

DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(
    pytorch_qnnp_requantize_precise__scalar_unsigned32)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(
    pytorch_qnnp_requantize_precise__scalar_unsigned64)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(
    pytorch_qnnp_requantize_precise__scalar_signed64)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__sse2)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__ssse3)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__sse4)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__neon)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__psimd)

DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__scalar_lrintf)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__scalar_magic)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__sse2)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__neon)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__psimd)

DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__scalar)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__sse2)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__ssse3)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__sse4)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__neon)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__psimd)

DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__scalar)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__sse2)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__ssse3)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__sse4)
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__neon)

#ifdef __cplusplus
} /* extern "C" */
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `stddef.h`
- `stdint.h`
- `qnnpack/params.h`
- `pthreadpool.h`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack`):

- [`log.h_docs.md`](./log.h_docs.md)
- [`x8zip.h_docs.md`](./x8zip.h_docs.md)
- [`AlignedAllocator.h_docs.md`](./AlignedAllocator.h_docs.md)
- [`requantization.h_docs.md`](./requantization.h_docs.md)
- [`pack.h_docs.md`](./pack.h_docs.md)
- [`common.h_docs.md`](./common.h_docs.md)
- [`u8maxpool.h_docs.md`](./u8maxpool.h_docs.md)
- [`assembly.h_docs.md`](./assembly.h_docs.md)
- [`q8gavgpool.h_docs.md`](./q8gavgpool.h_docs.md)
- [`sdwconv.h_docs.md`](./sdwconv.h_docs.md)


## Cross-References

- **File Documentation**: `requantization-stubs.h_docs.md`
- **Keyword Index**: `requantization-stubs.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack`):

- [`requantization-stubs.h_kw.md_docs.md`](./requantization-stubs.h_kw.md_docs.md)
- [`hgemm.h_docs.md_docs.md`](./hgemm.h_docs.md_docs.md)
- [`hgemm.h_kw.md_docs.md`](./hgemm.h_kw.md_docs.md)
- [`q8gemm.h_docs.md_docs.md`](./q8gemm.h_docs.md_docs.md)
- [`requantization.h_kw.md_docs.md`](./requantization.h_kw.md_docs.md)
- [`u8maxpool.h_docs.md_docs.md`](./u8maxpool.h_docs.md_docs.md)
- [`q8gavgpool.h_kw.md_docs.md`](./q8gavgpool.h_kw.md_docs.md)
- [`q8gemm.h_kw.md_docs.md`](./q8gemm.h_kw.md_docs.md)
- [`sgemm.h_docs.md_docs.md`](./sgemm.h_docs.md_docs.md)
- [`indirection.h_kw.md_docs.md`](./indirection.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `requantization-stubs.h_docs.md_docs.md`
- **Keyword Index**: `requantization-stubs.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
