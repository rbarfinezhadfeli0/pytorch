# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/fully-connected.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/fully-connected.c`
- **Size**: 4,827 bytes (4.71 KB)
- **Type**: Source File (.c)
- **Extension**: `.c`

## File Purpose

This is a source file (.c) that is part of the PyTorch project.

## Original Source

```c
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

enum pytorch_qnnp_status pytorch_qnnp_create_fully_connected_nc_q8(
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    pytorch_qnnp_operator_t* fully_connected_out) {
  pytorch_qnnp_operator_t fully_connected = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  for (int i = 0; i < output_channels; ++i) {
    if (requantization_scales[i] <= 0.0f ||
        !isnormal(requantization_scales[i])) {
      pytorch_qnnp_log_error(
          "failed to create fully connected operator with %.7g requantization scale: scale must be finite and positive",
          requantization_scales[i]);
      goto error;
    }
  }

  status = pytorch_qnnp_status_out_of_memory;

  fully_connected = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (fully_connected == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;

  const uint32_t n_stride = (output_channels + (nr - 1)) & -nr;
  const uint32_t k_stride = (input_channels + (kr - 1)) & -kr;

  fully_connected->packed_weights =
      malloc(n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
  if (fully_connected->packed_weights == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for packed weights",
        n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
    goto error;
  }
  memset(
      fully_connected->packed_weights,
      kernel_zero_points[0],
      n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));

  pytorch_pack_q8gemm_w(
      output_channels,
      input_channels,
      nr,
      nr,
      kr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
      input_zero_point,
      kernel_zero_points[0],
#endif
      kernel,
      bias,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
      kernel_zero_points,
#endif
      fully_connected->packed_weights);

  fully_connected->groups = 1;
  fully_connected->group_input_channels = input_channels;
  fully_connected->group_output_channels = output_channels;

  fully_connected->kernel_zero_point = kernel_zero_points[0];

  fully_connected->conv_quantization_params =
      pytorch_qnnp_compute_conv_quantization_params(
          input_zero_point,
          kernel_zero_points,
          requantization_scales,
          output_zero_point,
          output_min,
          output_max);

  fully_connected->ukernel_type = pytorch_qnnp_ukernel_type_gemm;
  fully_connected->format = pytorch_qnnp_format_quint8;

  *fully_connected_out = fully_connected;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(fully_connected);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_fully_connected_nc_q8(
    pytorch_qnnp_operator_t fully_connected,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    fully_connected->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  fully_connected->batch_size = 1;
  fully_connected->input_height = batch_size;
  fully_connected->input_width = 1;
  fully_connected->input = input;
  fully_connected->input_pixel_stride = input_stride;

  fully_connected->output_height = batch_size;
  fully_connected->output_width = 1;
  fully_connected->output = output;
  fully_connected->output_pixel_stride = output_stride;

  return pytorch_qnnp_status_success;
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/quantized/cpu/qnnpack/src`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src`):

- [`global-average-pooling.c_docs.md`](./global-average-pooling.c_docs.md)
- [`fully-connected-sparse.c_docs.md`](./fully-connected-sparse.c_docs.md)
- [`tanh.c_docs.md`](./tanh.c_docs.md)
- [`add.c_docs.md`](./add.c_docs.md)
- [`channel-shuffle.c_docs.md`](./channel-shuffle.c_docs.md)
- [`fc-dynamic-run.cc_docs.md`](./fc-dynamic-run.cc_docs.md)
- [`softargmax.c_docs.md`](./softargmax.c_docs.md)
- [`conv-run.cc_docs.md`](./conv-run.cc_docs.md)
- [`init.c_docs.md`](./init.c_docs.md)


## Cross-References

- **File Documentation**: `fully-connected.c_docs.md`
- **Keyword Index**: `fully-connected.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
