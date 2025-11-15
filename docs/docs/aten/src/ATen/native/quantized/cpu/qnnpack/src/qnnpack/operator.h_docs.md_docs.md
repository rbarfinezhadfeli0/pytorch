# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/operator.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/operator.h_docs.md`
- **Size**: 6,945 bytes (6.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/operator.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/operator.h`
- **Size**: 4,379 bytes (4.28 KB)
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

#include <qnnpack/requantization.h>

enum pytorch_qnnp_format {
  pytorch_qnnp_format_quint8 = 0x02000000,
  pytorch_qnnp_format_float32 = 0x02020202,
  pytorch_qnnp_format_float16 = 0x01010101,
};

enum pytorch_qnnp_ukernel_type {
  pytorch_qnnp_ukernel_type_none = 0,
  pytorch_qnnp_ukernel_type_add,
  pytorch_qnnp_ukernel_type_average_pooling,
  pytorch_qnnp_ukernel_type_channel_shuffle,
  pytorch_qnnp_ukernel_type_clamp,
  pytorch_qnnp_ukernel_type_conv,
  pytorch_qnnp_ukernel_type_dwconv,
  pytorch_qnnp_ukernel_type_gemm,
  pytorch_qnnp_ukernel_type_gemm_sparse_dq,
  pytorch_qnnp_ukernel_type_gemm_prepackA_sparse_dq,
  pytorch_qnnp_ukernel_type_global_average_pooling,
  pytorch_qnnp_ukernel_type_lut,
  pytorch_qnnp_ukernel_type_max_pooling,
  pytorch_qnnp_ukernel_type_softargmax,
  pytorch_qnnp_ukernel_type_xzp_gemm,
};

typedef struct {
  union {
    const uint32_t* col_indices_w32;
    const uint16_t* col_indices_w16;
    const uint8_t* col_indices_w8;
  };
  union {
    const uint32_t* row_values_w32;
    const uint16_t* row_values_w16;
    const uint8_t* row_values_w8;
  };
  const uint8_t* values;
  uint32_t row_block_size;
  uint32_t col_block_size;
  enum pytorch_qnnp_sparse_matrix_indices_dtype indices_dtype;
} sparse_matrix_t;

struct pytorch_qnnp_operator {
  size_t batch_size;
  uint32_t input_padding_depth;
  uint32_t input_padding_height;
  uint32_t input_padding_width;
  uint32_t adjustment_height;
  uint32_t adjustment_width;
  uint32_t kernel_depth;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t stride_depth;
  uint32_t stride_height;
  uint32_t stride_width;
  uint32_t dilation_depth;
  uint32_t dilation_height;
  uint32_t dilation_width;
  uint32_t groups;
  size_t group_stride;
  size_t group_channels;
  size_t group_input_channels;
  size_t group_output_channels;
  size_t channels;

  size_t input_depth;
  size_t input_height;
  size_t input_width;
  size_t input_pixel_stride;
  const void* input;
  const void** indirection_buffer;
  void* a_sum;

  size_t step_depth;
  size_t step_height;
  size_t step_width;

  size_t input2_pixel_stride;
  const void* input2;

  size_t output_depth;
  size_t output_height;
  size_t output_width;
  size_t output_pixel_stride;
  void* output;

  void* packed_weights;
  float input_scale;
  float output_scale;
  uint8_t input_zero_point;
  uint8_t kernel_zero_point;
  uint8_t output_zero_point;
  uint8_t output_min;
  uint8_t output_max;

  size_t valid_batch_size;
  size_t last_input_height;
  size_t last_input_width;
  const void* last_input;

  void* zero_buffer;
  void* zero_pointer;
  void* lookup_table;

  union {
    union pytorch_qnnp_q31_requantization_params requantization_params;
    union pytorch_qnnp_conv_quantization_params conv_quantization_params;
    union pytorch_qnnp_add_quantization_params add_quantization_params;
    union pytorch_qnnp_avgpool_quantization_params avgpool_quantization_params;
    union pytorch_qnnp_u8_clamping_params u8_clamping_params;
  };
  enum pytorch_qnnp_ukernel_type ukernel_type;
  enum pytorch_qnnp_format format;

  bool per_channel;
  bool transpose;

  // Sparsity support
  sparse_matrix_t sparse_matrix;
  const void* bias;
  struct pytorch_qnnp_conv_dynamic_quantization_params dynamic_conv_quantization_params;
  uint8_t* prepacked_a;
};

static inline uint32_t pytorch_qnnp_operator_get_log2_output_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  return (uint32_t)(convolution->format & UINT32_C(0xFF));
}

static inline uint32_t pytorch_qnnp_operator_get_log2_input_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  return (uint32_t)((convolution->format >> 8) & UINT32_C(0xFF));
}

static inline uint32_t pytorch_qnnp_operator_get_log2_kernel_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  return (uint32_t)((convolution->format >> 16) & UINT32_C(0xFF));
}

static inline uint32_t pytorch_qnnp_operator_get_log2_bias_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  return (uint32_t)((convolution->format >> 24) & UINT32_C(0xFF));
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `pytorch_qnnp_operator`, `pytorch_qnnp_conv_dynamic_quantization_params`, `pytorch_qnnp_operator`, `pytorch_qnnp_operator`, `pytorch_qnnp_operator`, `pytorch_qnnp_operator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `stddef.h`
- `stdint.h`
- `qnnpack/requantization.h`


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

- **File Documentation**: `operator.h_docs.md`
- **Keyword Index**: `operator.h_kw.md`
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

- **File Documentation**: `operator.h_docs.md_docs.md`
- **Keyword Index**: `operator.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
