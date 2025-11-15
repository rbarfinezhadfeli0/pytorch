# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8dwconv.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8dwconv.h_docs.md`
- **Size**: 6,003 bytes (5.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8dwconv.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8dwconv.h`
- **Size**: 3,619 bytes (3.53 KB)
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

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(                \
      size_t channels,                               \
      size_t output_width,                           \
      const uint8_t** input,                         \
      const void* weights,                           \
      uint8_t* output,                               \
      size_t input_stride,                           \
      size_t output_increment,                       \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_up8x9__neon)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_up8x9_per_channel__neon)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_up8x9__sse2)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2)

#define DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(                \
      size_t channels,                               \
      size_t output_width,                           \
      const uint8_t** input,                         \
      const void* weights,                           \
      int32_t* buffer,                               \
      uint8_t* output,                               \
      size_t input_stride,                           \
      size_t output_increment,                       \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_mp8x25__neon)
DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon)
DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_mp8x25__sse2)
DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2)

#define DECLARE_PYTORCH_Q8MPDWCONV_3D_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(                           \
      size_t channels,                                          \
      size_t output_height,                                     \
      size_t output_width,                                      \
      const uint8_t** input,                                    \
      const void* weights,                                      \
      int32_t* buffer,                                          \
      uint8_t* output,                                          \
      size_t input_row_stride,                                  \
      size_t input_col_stride,                                  \
      size_t output_increment,                                  \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8MPDWCONV_3D_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_mp8x27__neon)
DECLARE_PYTORCH_Q8MPDWCONV_3D_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_mp8x27__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

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
- `qnnpack/common.h`
- `qnnpack/params.h`


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

- **File Documentation**: `q8dwconv.h_docs.md`
- **Keyword Index**: `q8dwconv.h_kw.md`
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

- **File Documentation**: `q8dwconv.h_docs.md_docs.md`
- **Keyword Index**: `q8dwconv.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
