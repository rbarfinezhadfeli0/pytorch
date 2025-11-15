# Documentation: `docs/aten/src/ATen/native/quantized/cpu/XnnpackUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/XnnpackUtils.h_docs.md`
- **Size**: 16,640 bytes (16.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/XnnpackUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/XnnpackUtils.h`
- **Size**: 14,105 bytes (13.77 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#ifdef USE_XNNPACK
#include <cstdint>

#include <ATen/core/Tensor.h>
#include <ATen/native/xnnpack/Common.h>

using xnnpack_operator = at::native::xnnpack::Operator;

namespace at::native::xnnp_utils {

/*
 * Return shape in the same order as the memory format
 * e.g. channels_last will return NHWC instead of NCHW
 */
std::vector<size_t> get_mem_format_aware_shape(const at::Tensor& in);

/*
 * Input is always int8_t, output can be [int8_t, uint8_t].
 * input  + offset = output
 * int8_t + 128    = uint8_t
 * int8_t + 0      = int8_t
 */
template <typename PT>
void q8_copy_int8_weight_and_add_offset(const at::Tensor& in, at::Tensor& out);

template <int kSpatialDim>
Tensor convert_conv_weights_to_channel_last_tensor(
    const at::Tensor& src,
    int groups,
    bool transpose);

/*
 * Series of create wrapper functions to call xnn_create_[de]conv* functions.
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_create_convolution2d_nhwc(
    uint32_t pad_top,
    uint32_t pad_right,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t ip_chan_stride,
    size_t op_chan_stride,
    int8_t izp,
    float ip_scale,
    int8_t kzp,
    const float* k_scales,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t ozp,
    float op_scale,
    int8_t op_min,
    int8_t op_max,
    uint32_t flags,
    xnn_operator_t* op,
    bool per_channel,
    bool transpose) {
  /* Symmetric quantization forces kzp = 0 */
  TORCH_CHECK(!kzp, "XNNPACK Q[SC]8 conv kernels expects kernel zero point to be zero."
                    "But got: ", kzp);

  if (transpose) {
    TORCH_CHECK(!per_channel, "XNNPACK Q[SC]8 does not have a per channel deconvolution!");
    return xnn_create_deconvolution2d_nhwc_qs8(
        pad_top,        /* uint32_t output_padding_top          */
        pad_right,      /* uint32_t output_padding_right        */
        pad_bottom,     /* uint32_t output_padding_bottom       */
        pad_left,       /* uint32_t output_padding_left         */
        kernel_h,       /* uint32_t kernel_height               */
        kernel_w,       /* uint32_t kernel_width                */
        stride_h,       /* uint32_t stride_height               */
        stride_w,       /* uint32_t stride_width                */
        dilation_h,     /* uint32_t dilation_height             */
        dilation_w,     /* uint32_t dilation_width              */
        groups,         /* uint32_t groups                      */
        group_input_channels,  /* size_t group_input_channels   */
        group_output_channels, /* size_t group_output_channels  */
        ip_chan_stride, /* size_t input_pixel_stride            */
        op_chan_stride, /* size_t output_pixel_stride           */
        izp,            /* int8_t input_zero_point              */
        ip_scale,       /* float input_scale                    */
        k_scales[0],    /* float kernel_scale                   */
        kernel,         /* const int8_t* kernel                 */
        bias,           /* const int32_t* bias                  */
        ozp,            /* int8_t output_zero_point             */
        op_scale,       /* float output_scale                   */
        op_min,         /* int8_t output_min                    */
        op_max,         /* int8_t output_max                    */
        flags,          /* uint32_t flags                       */
        nullptr,        /* xnn_caches_t caches                  */
        nullptr,        /* xnn_weights_cache_t weights_cache    */
        op);            /* xnn_operator_t* deconvolution_op_out */

  }

  if (!per_channel) {
    return xnn_create_convolution2d_nhwc_qs8(
        pad_top,        /* uint32_t input_padding_top         */
        pad_right,      /* uint32_t input_padding_right       */
        pad_bottom,     /* uint32_t input_padding_bottom      */
        pad_left,       /* uint32_t input_padding_left        */
        kernel_h,       /* uint32_t kernel_height             */
        kernel_w,       /* uint32_t kernel_width              */
        stride_h,       /* uint32_t subsampling_height        */
        stride_w,       /* uint32_t subsampling_width         */
        dilation_h,     /* uint32_t dilation_height           */
        dilation_w,     /* uint32_t dilation_width            */
        groups,         /* uint32_t groups                    */
        group_input_channels,  /* size_t group_input_channels */
        group_output_channels, /* size_t group_output_channels*/
        ip_chan_stride, /* size_t input_channel_stride        */
        op_chan_stride, /* size_t output_channel_stride       */
        izp,            /* int8_t input_zero_point            */
        ip_scale,       /* float input_scale                  */
        k_scales[0],    /* float kernel_scale                 */
        kernel,         /* const int8_t* kernel               */
        bias,           /* const int32_t* bias                */
        ozp,            /* int8_t output_zero_point           */
        op_scale,       /* float output_scale                 */
        op_min,         /* int8_t output_min                  */
        op_max,         /* int8_t output_max                  */
        flags,          /* uint32_t flags                     */
        nullptr,        /* xnn_caches_t caches                */
        nullptr,        /* xnn_weights_cache_t weights_cache    */
        op);            /* xnn_operator_t* convolution_op_out */
  } else { /* per_channel */
    return xnn_create_convolution2d_nhwc_qs8_qc8w(
        pad_top,        /* uint32_t input_padding_top         */
        pad_right,      /* uint32_t input_padding_right       */
        pad_bottom,     /* uint32_t input_padding_bottom      */
        pad_left,       /* uint32_t input_padding_left        */
        kernel_h,       /* uint32_t kernel_height             */
        kernel_w,       /* uint32_t kernel_width              */
        stride_h,       /* uint32_t subsampling_height        */
        stride_w,       /* uint32_t subsampling_width         */
        dilation_h,     /* uint32_t dilation_height           */
        dilation_w,     /* uint32_t dilation_width            */
        groups,         /* uint32_t groups                    */
        group_input_channels,  /* size_t group_input_channels */
        group_output_channels, /* size_t group_output_channels*/
        ip_chan_stride, /* size_t input_channel_stride        */
        op_chan_stride, /* size_t output_channel_stride       */
        izp,            /* int8_t input_zero_point            */
        ip_scale,       /* float input_scale                  */
        k_scales,       /* const float* kernel_scale          */
        kernel,         /* const int8_t* kernel               */
        bias,           /* const int32_t* bias                */
        ozp,            /* int8_t output_zero_point           */
        op_scale,       /* float output_scale                 */
        op_min,         /* int8_t output_min                  */
        op_max,         /* int8_t output_max                  */
        flags,          /* uint32_t flags                     */
        nullptr,        /* xnn_caches_t caches                */
        nullptr,        /* xnn_weights_cache_t weights_cache    */
        op);            /* xnn_operator_t* convolution_op_out */
  }
}

/*
 * Series of reshape wrapper functions to call xnn_reshape_[de]conv* functions.
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_reshape_convolution2d_nhwc(
    xnn_operator_t op,
    size_t batch,
    size_t in_h,
    size_t in_w,
    pthreadpool_t pt_pool,
    bool per_channel = false,
    bool transpose = false,
    uint32_t adj_h = 0,
    uint32_t adj_w = 0) {
  if(transpose) {
    TORCH_CHECK(!per_channel, "XNNPACK Q[SC]8 does not have a per channel deconvolution!");
    return xnn_reshape_deconvolution2d_nhwc_qs8(
        op,       /* xnn_operator_t deconvolution_op */
        batch,    /* size_t batch_size               */
        in_h,     /* size_t input_height             */
        in_w,     /* size_t input_width              */
        adj_h,    /* uint32_t adjustment_height      */
        adj_w,    /* uint32_t adjustment_width       */
        nullptr,  /* size_t* output_height_out       */
        nullptr,  /* size_t* output_width_out        */
        pt_pool); /* pthreadpool_t threadpool        */
  }

  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;

  if (!per_channel) {
    return xnn_reshape_convolution2d_nhwc_qs8(
        op,       /* xnn_operator_t convolution_op */
        batch,    /* size_t batch_size             */
        in_h,     /* size_t input_height           */
        in_w,     /* size_t input_width            */
        &workspace_size, /* size_t* workspace_size */
        &workspace_alignment, /* size_t* workspace_alignment */
        nullptr,  /* size_t* output_height_out     */
        nullptr,  /* size_t* output_width_out      */
        pt_pool); /* pthreadpool_t threadpool      */
  } else { /* per_channel */
    return xnn_reshape_convolution2d_nhwc_qs8_qc8w(
        op,       /* xnn_operator_t convolution_op */
        batch,    /* size_t batch_size             */
        in_h,     /* size_t input_height           */
        in_w,     /* size_t input_width            */
        &workspace_size, /* size_t* workspace_size */
        &workspace_alignment, /* size_t* workspace_alignment */
        nullptr,  /* size_t* output_height_out     */
        nullptr,  /* size_t* output_width_out      */
        pt_pool); /* pthreadpool_t threadpool      */
  }
}


/*
 * Series of setup wrapper functions to call xnn_setup_[de]conv* functions.
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_setup_convolution2d_nhwc(
    xnn_operator_t op,
    const int8_t* inp,
    int8_t* outp,
    bool per_channel = false,
    bool transpose = false) {
  if(transpose) {
    TORCH_CHECK(!per_channel, "XNNPACK Q[SC]8 does not have a per channel deconvolution!");

    return xnn_setup_deconvolution2d_nhwc_qs8(
        op,       /* xnn_operator_t deconvolution_op */
        inp,      /* const int8_t* input             */
        outp);    /* int8_t* output                  */
  }

  if (!per_channel) {
    return xnn_setup_convolution2d_nhwc_qs8(
        op,       /* xnn_operator_t deconvolution_op */
        nullptr,  /* void workspace                  */
        inp,      /* const int8_t* input             */
        outp);    /* int8_t* output                  */
  } else { /* per_channel */
    return xnn_setup_convolution2d_nhwc_qs8_qc8w(
        op,       /* xnn_operator_t deconvolution_op */
        nullptr,  /* void workspace                  */
        inp,      /* const int8_t* input             */
        outp);    /* int8_t* output                  */
  }
}


/*
 * Series of wrapper functions to call xnn_create* and xnn_setup*
 * functions for linear
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_create_fully_connected_nc(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    int8_t input_zero_point,
    float input_scale,
    int8_t kernel_zero_point,
    float kernel_scale,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* fully_connected_op_out) {
  /* Symmetric quantization forces kzp = 0 */
  TORCH_CHECK(!kernel_zero_point, "XNNPACK QS8 linear kernel expects kernel zero point to be zero."
                    "But got: ", kernel_zero_point);
  return xnn_create_fully_connected_nc_qs8(
      input_channels,          /* size_t input_channels                  */
      output_channels,         /* size_t output_channels                 */
      input_stride,            /* size_t input_stride                    */
      output_stride,           /* size_t output_stride                   */
      input_zero_point,        /* int8_t input_zero_point                */
      input_scale,             /* float input_scale                      */
      kernel_scale,            /* float kernel_scale                     */
      kernel,                  /* const int8_t* kernel                   */
      bias,                    /* const int32_t* bias                    */
      output_zero_point,       /* int8_t output_zero_point               */
      output_scale,            /* float output_scale                     */
      output_min,              /* int8_t output_min                      */
      output_max,              /* int8_t output_max                      */
      flags,                   /* uint32_t flags                         */
      nullptr,                 /* xnn_caches_t caches                    */
      nullptr,                 /* xnn_weights_cache_t                    */
      fully_connected_op_out); /* xnn_operator_t* fully_connected_op_out */
}

C10_ALWAYS_INLINE
enum xnn_status xnnp_reshape_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    pthreadpool_t threadpool) {
  return xnn_reshape_fully_connected_nc_qs8(
      fully_connected_op, /* xnn_operator_t fully_connected_op */
      batch_size,         /* size_t batch_size                 */
      threadpool);        /* pthreadpool_t threadpool          */
}

C10_ALWAYS_INLINE
enum xnn_status xnnp_setup_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    const int8_t* input,
    int8_t* output) {
  return xnn_setup_fully_connected_nc_qs8(
      fully_connected_op, /* xnn_operator_t fully_connected_op */
      input,              /* const int8_t* input               */
      output              /* int8_t* output                    */
    );
}

} // namespace at::native::xnnp_utils

#endif // USE_XNNPACK

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `ATen/core/Tensor.h`
- `ATen/native/xnnpack/Common.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `XnnpackUtils.h_docs.md`
- **Keyword Index**: `XnnpackUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu`):

- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`init_qnnpack.cpp_docs.md_docs.md`](./init_qnnpack.cpp_docs.md_docs.md)
- [`qelu.cpp_kw.md_docs.md`](./qelu.cpp_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)
- [`qclamp.cpp_docs.md_docs.md`](./qclamp.cpp_docs.md_docs.md)
- [`qembeddingbag_prepack.h_docs.md_docs.md`](./qembeddingbag_prepack.h_docs.md_docs.md)
- [`qdropout.cpp_docs.md_docs.md`](./qdropout.cpp_docs.md_docs.md)
- [`qelu.cpp_docs.md_docs.md`](./qelu.cpp_docs.md_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md_docs.md`](./qembeddingbag_unpack.cpp_docs.md_docs.md)
- [`LinearUnpackImpl.cpp_kw.md_docs.md`](./LinearUnpackImpl.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `XnnpackUtils.h_docs.md_docs.md`
- **Keyword Index**: `XnnpackUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
