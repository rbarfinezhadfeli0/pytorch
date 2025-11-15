# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/channel-shuffle.c`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/channel-shuffle.c`
- **Size**: 2,998 bytes (2.93 KB)
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

enum pytorch_qnnp_status pytorch_qnnp_create_channel_shuffle_nc_x8(
    size_t groups,
    size_t group_channels,
    uint32_t flags,
    pytorch_qnnp_operator_t* channel_shuffle_out) {
  pytorch_qnnp_operator_t channel_shuffle_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_channel_shuffle_nc_x8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (groups <= 1) {
    pytorch_qnnp_log_error(
        "failed to create channel shuffle operator with %zu groups: "
        "at least two groups required",
        groups);
    goto error;
  }

  if (group_channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create channel shuffle operator with %zu group channels: "
        "number of group channels must be non-zero",
        group_channels);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  channel_shuffle_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (channel_shuffle_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  channel_shuffle_op->groups = groups;
  channel_shuffle_op->group_channels = group_channels;

  channel_shuffle_op->ukernel_type = pytorch_qnnp_ukernel_type_channel_shuffle;
  channel_shuffle_op->format = pytorch_qnnp_format_quint8;

  *channel_shuffle_out = channel_shuffle_op;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(channel_shuffle_op);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_channel_shuffle_nc_x8(
    pytorch_qnnp_operator_t channel_shuffle_op,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_channel_shuffle_nc_x8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    channel_shuffle_op->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  channel_shuffle_op->batch_size = batch_size;
  channel_shuffle_op->input = input;
  channel_shuffle_op->input_pixel_stride = input_stride;
  channel_shuffle_op->output = output;
  channel_shuffle_op->output_pixel_stride = output_stride;

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
- [`fc-dynamic-run.cc_docs.md`](./fc-dynamic-run.cc_docs.md)
- [`softargmax.c_docs.md`](./softargmax.c_docs.md)
- [`fully-connected.c_docs.md`](./fully-connected.c_docs.md)
- [`conv-run.cc_docs.md`](./conv-run.cc_docs.md)
- [`init.c_docs.md`](./init.c_docs.md)


## Cross-References

- **File Documentation**: `channel-shuffle.c_docs.md`
- **Keyword Index**: `channel-shuffle.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
