# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/global-average-pooling.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/global-average-pooling.c`
- **Size**: 5,450 bytes (5.32 KB)
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
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

enum pytorch_qnnp_status pytorch_qnnp_create_global_average_pooling_nwc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* global_average_pooling_out) {
  pytorch_qnnp_operator_t global_average_pooling_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_global_average_pooling_nwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create global average pooling operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create global average pooling operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create global average pooling operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    pytorch_qnnp_log_error(
        "failed to create global average pooling operator with %.7g input-to-output scale ratio: "
        "scale ratio must be in [2**-8, 2**8) range",
        input_output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  global_average_pooling_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (global_average_pooling_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  void* zero_buffer = calloc(channels, sizeof(uint8_t));
  if (zero_buffer == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for zero padding",
        channels * sizeof(uint8_t));
    goto error;
  }
  global_average_pooling_op->zero_buffer = zero_buffer;
  global_average_pooling_op->zero_pointer = zero_buffer;

  global_average_pooling_op->channels = channels;
  global_average_pooling_op->input_zero_point = input_zero_point;
  global_average_pooling_op->output_zero_point = output_zero_point;
  global_average_pooling_op->input_scale = input_scale;
  global_average_pooling_op->output_scale = output_scale;
  global_average_pooling_op->output_min = output_min;
  global_average_pooling_op->output_max = output_max;

  global_average_pooling_op->ukernel_type =
      pytorch_qnnp_ukernel_type_global_average_pooling;
  global_average_pooling_op->format = pytorch_qnnp_format_quint8;

  *global_average_pooling_out = global_average_pooling_op;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(global_average_pooling_op);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_global_average_pooling_nwc_q8(
    pytorch_qnnp_operator_t global_average_pooling_op,
    size_t batch_size,
    size_t width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_global_average_pooling_nwc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    global_average_pooling_op->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  if (width == 0) {
    pytorch_qnnp_log_error(
        "failed to setup global average pooling operator with width %zu: width must be non-zero",
        width);
    return pytorch_qnnp_status_invalid_parameter;
  }

  global_average_pooling_op->batch_size = batch_size;
  global_average_pooling_op->input_width = width;
  global_average_pooling_op->input = input;
  global_average_pooling_op->input_pixel_stride = input_stride;
  global_average_pooling_op->output = output;
  global_average_pooling_op->output_pixel_stride = output_stride;

  global_average_pooling_op->avgpool_quantization_params =
      pytorch_qnnp_compute_avgpool_quantization_params(
          -(int32_t)width *
              (int32_t)(uint32_t)global_average_pooling_op->input_zero_point,
          global_average_pooling_op->input_scale /
              (global_average_pooling_op->output_scale * (float)width),
          global_average_pooling_op->output_zero_point,
          global_average_pooling_op->output_min,
          global_average_pooling_op->output_max);

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

- [`fully-connected-sparse.c_docs.md`](./fully-connected-sparse.c_docs.md)
- [`tanh.c_docs.md`](./tanh.c_docs.md)
- [`add.c_docs.md`](./add.c_docs.md)
- [`channel-shuffle.c_docs.md`](./channel-shuffle.c_docs.md)
- [`fc-dynamic-run.cc_docs.md`](./fc-dynamic-run.cc_docs.md)
- [`softargmax.c_docs.md`](./softargmax.c_docs.md)
- [`fully-connected.c_docs.md`](./fully-connected.c_docs.md)
- [`conv-run.cc_docs.md`](./conv-run.cc_docs.md)
- [`init.c_docs.md`](./init.c_docs.md)


## Cross-References

- **File Documentation**: `global-average-pooling.c_docs.md`
- **Keyword Index**: `global-average-pooling.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
