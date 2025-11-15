# Documentation: q8gemm_sparse.h

## File Metadata
- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8gemm_sparse.h`
- **Size**: 7636 bytes
- **Lines**: 150
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
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

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Structures
This file defines 2 struct(s): pytorch_qnnp_conv_dynamic_quantization_params, pytorch_qnnp_conv_dynamic_quantization_params


## Key Components

The file contains 321 words across 150 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 7636 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
