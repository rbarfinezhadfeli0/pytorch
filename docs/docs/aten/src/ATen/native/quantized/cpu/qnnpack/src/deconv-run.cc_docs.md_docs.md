# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/deconv-run.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/deconv-run.cc_docs.md`
- **Size**: 9,351 bytes (9.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/deconv-run.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/deconv-run.cc`
- **Size**: 6,632 bytes (6.48 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <qnnpack/indirection.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/params.h>

#include <cstring>
#include <memory>

namespace qnnpack {
namespace {
static size_t compute_output_dimension(
    size_t input_dimension,
    size_t input_padding_dimension,
    size_t adjustment_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t stride_dimension) {
  const size_t effective_kernel_dimension =
      (kernel_dimension - 1) * dilation_dimension + 1;
  return stride_dimension * (input_dimension - 1) + adjustment_dimension +
      effective_kernel_dimension - input_padding_dimension;
}
} // namespace

struct q8conv_context {
  size_t bs;
  size_t ks;
  size_t kc;
  size_t kc_stride;
  size_t m;
  size_t m_stride;
  size_t n;
  size_t n_stride;
  const uint8_t** indirect_a;
  const void* packed_w;
  uint8_t* c;
  size_t c_stride;
  union pytorch_qnnp_conv_quantization_params quantization_params;
  const pytorch_q8conv_ukernel_function ukernel;
};

static void compute_q8conv(
    const struct q8conv_context context[1],
    size_t group_index,
    size_t image_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t image_range /* always 1 */,
    size_t mr_block_size,
    size_t nr_block_size) {
  const size_t bs = context->bs;
  const size_t ks = context->ks;
  const size_t kc = context->kc;
  const size_t kc_stride = context->kc_stride;
  const size_t m = context->m;
  const size_t m_stride = context->m_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t** indirect_a = context->indirect_a;
  const void* packed_w = context->packed_w;
  uint8_t* c = context->c;
  const size_t c_stride = context->c_stride;

  const size_t output_channel_index = group_index * n + nr_block_start;
  context->ukernel(
      mr_block_size,
      nr_block_size,
      kc,
      ks,
      indirect_a +
          (mr_block_start + (image_index + group_index * bs) * m_stride) * ks,
      (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (kc_stride * sizeof(uint8_t) + sizeof(int32_t))),
      c + (mr_block_start + image_index * m) * c_stride + group_index * n +
          nr_block_start,
      c_stride,
      output_channel_index,
      &context->quantization_params);
};

struct QnnpackDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);
  }
};

enum pytorch_qnnp_status qnnpackDeConv(
    const pytorch_qnnp_operator_t deconvolution,
    void* packed_weights,
    const size_t batch_size,
    const size_t input_height,
    const size_t input_width,
    const uint8_t input_zero_point,
    const uint8_t* input,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    const uint8_t output_zero_point,
    const uint8_t output_min,
    const uint8_t output_max,
    uint8_t* output,
    pthreadpool_t threadpool) {

  if (batch_size == 0) {
    // Doesn't matter what's going on, if no batches, return
    return pytorch_qnnp_status_success;
  }
  // Check all invalid parameters
  const size_t kernel_width = deconvolution->kernel_width;
  const size_t kernel_height = deconvolution->kernel_height;

  // Support vars
  const size_t group_input_channels = deconvolution->group_input_channels;
  const size_t group_output_channels = deconvolution->group_output_channels;
  const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
  const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
  const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

  deconvolution->conv_quantization_params =
      pytorch_qnnp_compute_conv_quantization_params(
          input_zero_point,
          kernel_zero_points,
          requantization_scales,
          output_zero_point,
          output_min,
          output_max);

  // Setup the kernel
  const size_t output_width = compute_output_dimension(
      input_width,
      deconvolution->input_padding_width * 2,
      deconvolution->adjustment_width,
      kernel_width,
      deconvolution->dilation_width,
      deconvolution->stride_width);
  const size_t output_height = compute_output_dimension(
      input_height,
      deconvolution->input_padding_height * 2,
      deconvolution->adjustment_height,
      kernel_height,
      deconvolution->dilation_height,
      deconvolution->stride_height);
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_size = output_height * output_width;

  const size_t input_pixel_stride =
      deconvolution->group_input_channels * deconvolution->groups;
  const size_t output_pixel_stride =
      deconvolution->group_output_channels * deconvolution->groups;

  if (deconvolution->input != input ||
      deconvolution->batch_size != batch_size ||
      deconvolution->input_height != input_height ||
      deconvolution->input_width != input_width ||
      deconvolution->input_pixel_stride != input_pixel_stride) {
    pytorch_qnnp_status status = pytorch_qnnp_setup_deconvolution2d_nhwc_q8(
        deconvolution,
        batch_size,
        input_height,
        input_width,
        input,
        input_pixel_stride,
        output,
        output_pixel_stride,
        threadpool);
    if (status != pytorch_qnnp_status_success) {
      pytorch_qnnp_log_error(
          "failed to run deconvolution op setup to setup indirection buffer.");
      return status;
    }
  }

  // Run the kernel
  const size_t m_stride = round_up(output_size, mr);
  struct q8conv_context q8conv_context = {
      .bs = deconvolution->batch_size,
      .ks = kernel_size,
      .kc = group_input_channels,
      .kc_stride = k_stride * kernel_size,
      .m = output_size,
      .m_stride = m_stride,
      .n = group_output_channels,
      .n_stride = n_stride,
      .indirect_a = (const uint8_t**)deconvolution->indirection_buffer,
      .packed_w = packed_weights,
      .c = output,
      .c_stride = deconvolution->output_pixel_stride,
      .quantization_params = deconvolution->conv_quantization_params,
      .ukernel = pytorch_qnnp_params.q8conv.conv,
  };

  pthreadpool_compute_4d_tiled(
      threadpool,
      (pthreadpool_function_4d_tiled_t)compute_q8conv,
      &q8conv_context,
      deconvolution->groups,
      batch_size,
      output_size,
      group_output_channels,
      1,
      1,
      mr,
      nr);
  return pytorch_qnnp_status_success;
}
}  // namespace qnnpack

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `qnnpack`, `struct`

**Classes/Structs**: `q8conv_context`, `q8conv_context`, `QnnpackDeleter`, `q8conv_context`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `pytorch_qnnpack.h`
- `qnnpack_func.h`
- `qnnpack/indirection.h`
- `qnnpack/log.h`
- `qnnpack/math.h`
- `qnnpack/params.h`
- `cstring`
- `memory`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `deconv-run.cc_docs.md`
- **Keyword Index**: `deconv-run.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/src`):

- [`hardsigmoid.c_kw.md_docs.md`](./hardsigmoid.c_kw.md_docs.md)
- [`indirection.c_docs.md_docs.md`](./indirection.c_docs.md_docs.md)
- [`conv-prepack.cc_kw.md_docs.md`](./conv-prepack.cc_kw.md_docs.md)
- [`deconvolution.c_docs.md_docs.md`](./deconvolution.c_docs.md_docs.md)
- [`fully-connected.c_docs.md_docs.md`](./fully-connected.c_docs.md_docs.md)
- [`fully-connected-sparse.c_docs.md_docs.md`](./fully-connected-sparse.c_docs.md_docs.md)
- [`softargmax.c_kw.md_docs.md`](./softargmax.c_kw.md_docs.md)
- [`operator-run.c_docs.md_docs.md`](./operator-run.c_docs.md_docs.md)
- [`indirection.c_kw.md_docs.md`](./indirection.c_kw.md_docs.md)
- [`tanh.c_docs.md_docs.md`](./tanh.c_docs.md_docs.md)


## Cross-References

- **File Documentation**: `deconv-run.cc_docs.md_docs.md`
- **Keyword Index**: `deconv-run.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
