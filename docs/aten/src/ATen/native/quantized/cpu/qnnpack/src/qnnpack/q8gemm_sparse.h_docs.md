# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8gemm_sparse.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8gemm_sparse.h`
- **Size**: 7,636 bytes (7.46 KB)
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

#define DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const uint8_t* packed_w,                   \
      const uint32_t* w_row_ptr,                 \
      const uint32_t* w_block_ids_ptr,           \
      uint8_t* c,                                \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x8__neon)
DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_8x8__neon)

DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x8__aarch32_neon)

DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_8x8__aarch64_neon)

DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x4c2__sse2)

#define DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const uint8_t* packed_w,                   \
      const uint32_t* w_row_ptr,                 \
      const uint32_t* w_block_ids_ptr,           \
      const float* b,                            \
      float* c,                                  \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch64_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2)

#define DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION( \
    fn_name, w_index_dtype)                                                          \
  PYTORCH_QNNP_INTERNAL void fn_name(                                                \
      size_t mr,                                                                     \
      size_t nr,                                                                     \
      const uint8_t* a_packed,                                                       \
      const uint8_t* packed_w,                                                       \
      const w_index_dtype* w_row_ptr,                                                \
      const w_index_dtype* w_block_ids_ptr,                                          \
      const float* b,                                                                \
      float* c,                                                                      \
      size_t c_stride,                                                               \
      size_t output_channel_index,                                                   \
      const struct pytorch_qnnp_conv_dynamic_quantization_params*                    \
          quantization_params);

// w32, w16, and w8 refer to variants of the kernel which use uint32_t,
// uint16_t, and uint8_t datatype for row values/col indices respectively
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w32__aarch32_neon,
    uint32_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w16__aarch32_neon,
    uint16_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w8__aarch32_neon,
    uint8_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w32__aarch32_neon,
    uint32_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w16__aarch32_neon,
    uint16_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w8__aarch32_neon,
    uint8_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon,
    uint32_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w32__aarch64_neon,
    uint32_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w16__aarch64_neon,
    uint16_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w8__aarch64_neon,
    uint8_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w32__aarch64_neon,
    uint32_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w16__aarch64_neon,
    uint16_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w8__aarch64_neon,
    uint8_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2,
    uint32_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2,
    uint16_t)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2,
    uint8_t)

#define DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      const size_t mr,                           \
      const size_t K,                            \
      const uint8_t* a,                          \
      const size_t a_stride,                     \
      uint8_t* a_packed);

DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon)
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `pytorch_qnnp_conv_dynamic_quantization_params`, `pytorch_qnnp_conv_dynamic_quantization_params`


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

- **File Documentation**: `q8gemm_sparse.h_docs.md`
- **Keyword Index**: `q8gemm_sparse.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
