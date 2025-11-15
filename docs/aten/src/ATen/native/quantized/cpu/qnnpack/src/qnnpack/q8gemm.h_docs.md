# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8gemm.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8gemm.h`
- **Size**: 4,025 bytes (3.93 KB)
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

#define DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      size_t k,                                  \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const void* w,                             \
      uint8_t* c,                                \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_3x3c8__neon)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_2x4c8__neon)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x8__neon)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_6x4__neon)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_8x8__neon)

DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x8__aarch32_neon)

DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_8x8__aarch64_neon)

DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_2x4c8__sse2)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x4c2__sse2)

#define DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      size_t k,                                  \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const void* w,                             \
      const float* b,                            \
      float* c,                                  \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(pytorch_q8gemm_dq_ukernel_4x8__neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(pytorch_q8gemm_dq_ukernel_4x4c2__sse2)

#define DECLARE_PYTORCH_Q8GEMM_XZP_UKERNEL_FUNCTION(fn_name)      \
  PYTORCH_QNNP_INTERNAL void fn_name(                     \
      size_t mr,                                          \
      size_t nr,                                          \
      size_t k,                                           \
      const uint8_t* a,                                   \
      size_t a_stride,                                    \
      const int32_t* a_sum,                               \
      const void* w,                                      \
      uint8_t* c,                                         \
      size_t c_stride,                                    \
      const union pytorch_qnnp_q31_requantization_params* \
          requantization_params);
DECLARE_PYTORCH_Q8GEMM_XZP_UKERNEL_FUNCTION(pytorch_q8gemm_xzp_ukernel_4x8c2__neon)
DECLARE_PYTORCH_Q8GEMM_XZP_UKERNEL_FUNCTION(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon)

PYTORCH_QNNP_INTERNAL void pytorch_q8sumrows_ukernel_4x__neon(
    const uint8_t* a,
    size_t m,
    size_t k,
    size_t stride,
    const int32_t multiplier,
    int32_t* row_sum);

#ifdef __cplusplus
} /* extern "C" */
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `pytorch_qnnp_conv_dynamic_quantization_params`


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

- **File Documentation**: `q8gemm.h_docs.md`
- **Keyword Index**: `q8gemm.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
