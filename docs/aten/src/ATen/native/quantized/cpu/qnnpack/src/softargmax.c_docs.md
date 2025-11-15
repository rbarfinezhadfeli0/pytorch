# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/softargmax.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/softargmax.c`
- **Size**: 4,165 bytes (4.07 KB)
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

enum pytorch_qnnp_status pytorch_qnnp_create_softargmax_nc_q8(
    size_t channels,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint32_t flags,
    pytorch_qnnp_operator_t* softargmax_out) {
  pytorch_qnnp_operator_t softargmax_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_softargmax_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  if (output_scale != 0x1.0p-8f) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g output scale: only output scale of 1/256 is supported",
        output_scale);
    goto error;
  }

  if (output_zero_point != 0) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %" PRIu8
        " output zero point: only output zero point of 0 is supported",
        output_zero_point);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  softargmax_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (softargmax_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  softargmax_op->lookup_table = malloc(256 * sizeof(uint32_t));
  if (softargmax_op->lookup_table == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate 256 bytes for Soft ArgMax lookup table");
    goto error;
  }

  uint32_t* lookup_table = softargmax_op->lookup_table;
  const double qscale =
      fmin(((double)UINT32_MAX) / (double)channels, 8388607.0);
  for (int32_t i = 0; i < 256; i++) {
    const double scaled_exp_xi =
        qscale * exp((double)(i - 255) * (double)input_scale);
    lookup_table[(uint32_t)i] = (uint32_t)lrint(scaled_exp_xi);
  }

  softargmax_op->channels = channels;

  softargmax_op->ukernel_type = pytorch_qnnp_ukernel_type_softargmax;
  softargmax_op->format = pytorch_qnnp_format_quint8;

  *softargmax_out = softargmax_op;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(softargmax_op);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_softargmax_nc_q8(
    pytorch_qnnp_operator_t softargmax,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_softargmax_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    softargmax->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  softargmax->batch_size = batch_size;
  softargmax->input = input;
  softargmax->input_pixel_stride = input_stride;
  softargmax->output = output;
  softargmax->output_pixel_stride = output_stride;

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
- [`fully-connected.c_docs.md`](./fully-connected.c_docs.md)
- [`conv-run.cc_docs.md`](./conv-run.cc_docs.md)
- [`init.c_docs.md`](./init.c_docs.md)


## Cross-References

- **File Documentation**: `softargmax.c_docs.md`
- **Keyword Index**: `softargmax.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
