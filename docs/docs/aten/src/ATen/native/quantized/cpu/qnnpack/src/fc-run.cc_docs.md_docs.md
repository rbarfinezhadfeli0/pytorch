# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-run.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-run.cc_docs.md`
- **Size**: 6,340 bytes (6.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-run.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-run.cc`
- **Size**: 3,760 bytes (3.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <cstring>

namespace qnnpack {
struct q8gemm_context {
  size_t k;
  size_t k_stride;
  size_t n;
  size_t n_stride;
  const uint8_t* a;
  size_t a_stride;
  const uint8_t* packed_w;
  uint8_t* c;
  size_t c_stride;
  union pytorch_qnnp_conv_quantization_params quantization_params;
  const pytorch_q8gemm_ukernel_function ukernel;
};

static void compute_q8gemm(
    const struct q8gemm_context context[1],
    size_t group_index,
    size_t pixel_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t pixel_range,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t k = context->k;
  const size_t k_stride = context->k_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t* a = context->a;
  const size_t a_stride = context->a_stride;
  const void* packed_w = context->packed_w;
  uint8_t* c = context->c;
  const size_t c_stride = context->c_stride;

  size_t output_channel_index = nr_block_start;
  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,
      a_stride,
      (const void*) ((uintptr_t) packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(uint8_t) + sizeof(int32_t))),
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start + group_index * n,
      c_stride,
      output_channel_index,
      &context->quantization_params);
}

enum pytorch_qnnp_status qnnpackLinear(
    const size_t batch_size,
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    const uint8_t output_zero_point,
    const uint8_t output_min,
    const uint8_t output_max,
    const uint8_t* input,
    const size_t input_stride,
    void* packed_weights,
    uint8_t* output,
    const size_t output_stride,
    pthreadpool_t threadpool)
{
  const size_t groups = 1;
  const size_t group_input_channels = input_channels;
  const size_t group_output_channels = output_channels;
  const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
  const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
  const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

  const size_t output_size = batch_size * 1;
  union pytorch_qnnp_conv_quantization_params conv_quantization_params =
      pytorch_qnnp_compute_conv_quantization_params(
          input_zero_point, kernel_zero_points,
          requantization_scales, output_zero_point, output_min, output_max);

  struct q8gemm_context q8gemm_context = {
      .k = group_input_channels,
      .k_stride = k_stride,
      .n = group_output_channels,
      .n_stride = n_stride,
      .a = input,
      .a_stride = input_stride,
      .packed_w = (uint8_t*) packed_weights,
      .c = output,
      .c_stride = output_stride,
      .quantization_params = conv_quantization_params,
      .ukernel = pytorch_qnnp_params.q8conv.gemm,
  };

  if (output_size == 0) {
      // pthreadpool can tolerate a range of 0, but not a tile of 0.
      // We use output_size as a tile size, so bail here if it's 0.
      return pytorch_qnnp_status_success;
  }

  pthreadpool_compute_4d_tiled(
      threadpool,
      (pthreadpool_function_4d_tiled_t) compute_q8gemm,
      &q8gemm_context,
      groups,
      1 * output_size,
      output_size,
      group_output_channels,
      1,
      output_size,
      mr,
      nr);

  return pytorch_qnnp_status_success;
}
} // namespace qnnpack

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `qnnpack`

**Classes/Structs**: `q8gemm_context`, `q8gemm_context`, `q8gemm_context`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `pytorch_qnnpack.h`
- `qnnpack_func.h`
- `cstring`


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

- **File Documentation**: `fc-run.cc_docs.md`
- **Keyword Index**: `fc-run.cc_kw.md`
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

- **File Documentation**: `fc-run.cc_docs.md_docs.md`
- **Keyword Index**: `fc-run.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
