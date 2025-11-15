# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/leaky-relu.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/leaky-relu.c`
- **Size**: 5,225 bytes (5.10 KB)
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

enum pytorch_qnnp_status pytorch_qnnp_create_leaky_relu_nc_q8(
    size_t channels,
    float negative_slope,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* leaky_relu_out) {
  pytorch_qnnp_operator_t leaky_relu_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_leaky_relu_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (negative_slope <= 0.0f || !isnormal(negative_slope)) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g negative slope: slope must be finite and positive",
        negative_slope);
    goto error;
  }

  if (negative_slope > 1.0f) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g negative slope: slope must not exceed 1.0",
        negative_slope);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g input-to-output scale ratio: "
        "scale ratio must be in [2**-8, 2**8) range",
        input_output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  leaky_relu_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (leaky_relu_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  leaky_relu_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (leaky_relu_op->lookup_table == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate 256 bytes for Leaky ReLU lookup table");
    goto error;
  }

  uint8_t* lookup_table = leaky_relu_op->lookup_table;
  const float scaled_min_less_zero_point =
      (float)((int32_t)output_min - (int32_t)output_zero_point);
  const float scaled_max_less_zero_point =
      (float)((int32_t)output_max - (int32_t)output_zero_point);
  for (int32_t i = 0; i < 256; i++) {
    const float x =
        input_output_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    float y = x < 0.0f ? x * negative_slope : x;
    if (y < scaled_min_less_zero_point) {
      y = scaled_min_less_zero_point;
    }
    if (y > scaled_max_less_zero_point) {
      y = scaled_max_less_zero_point;
    }
    lookup_table[(uint32_t)i] = (uint8_t)(lrintf(y) + (long)output_zero_point);
  }

  leaky_relu_op->channels = channels;

  leaky_relu_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
  leaky_relu_op->format = pytorch_qnnp_format_quint8;

  *leaky_relu_out = leaky_relu_op;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(leaky_relu_op);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_leaky_relu_nc_q8(
    pytorch_qnnp_operator_t leaky_relu,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_leaky_relu_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    leaky_relu->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  leaky_relu->batch_size = batch_size;
  leaky_relu->input = input;
  leaky_relu->input_pixel_stride = input_stride;
  leaky_relu->output = output;
  leaky_relu->output_pixel_stride = output_stride;

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
- [`fully-connected.c_docs.md`](./fully-connected.c_docs.md)
- [`conv-run.cc_docs.md`](./conv-run.cc_docs.md)
- [`init.c_docs.md`](./init.c_docs.md)


## Cross-References

- **File Documentation**: `leaky-relu.c_docs.md`
- **Keyword Index**: `leaky-relu.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
