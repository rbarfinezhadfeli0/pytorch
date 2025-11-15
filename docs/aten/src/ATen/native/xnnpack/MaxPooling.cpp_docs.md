# Documentation: `aten/src/ATen/native/xnnpack/MaxPooling.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/xnnpack/MaxPooling.cpp`
- **Size**: 10,451 bytes (10.21 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_XNNPACK

#include <ATen/native/Pool.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/xnnpack/Pooling.h>

namespace at::native::xnnpack {

// Supports NHWC and NCHW FP32 max pooling with any
//  - kernel size
//  - padding
//  - stride
//  - dilation

bool use_max_pool2d(
    const Tensor& input,
    const IntArrayRef kernel_,
    const IntArrayRef padding_,
    IntArrayRef stride_,
    const IntArrayRef dilation_,
    const bool ceil_mode,
    const float output_min,
    const float output_max) {
  using namespace internal;

  // Make sure we are not dealing with an unorthodox configuration.
  if (kernel_.empty() || padding_.empty() || dilation_.empty()) {
    return false;
  }

  // Stride can be legitimately empty, in which case it is to be defaulted to kernel size.
  if (stride_.empty()) {
    stride_ = kernel_;
  }

  // Normalize the parameters.
  const internal::pooling::Parameters parameters{
    kernel_,
    padding_,
    stride_,
    dilation_,
  };

  // Here are the list of conditions required for this code path to be taken:
  // * Input must be 4D CPU float tensor with no gradients.
  // * Kernel must be a 2D IntArrayRef containing two positive numbers.
  //   Furthermore, 1x1 kernels are not valid as XNNPACK prohibits their use.
  // * Padding must be a 2D IntArrayRef containing two non-negative numbers.
  // * Stride must be a 2D IntArrayRef containing two positive numbers.
  // * Dilation must be a 2D IntArrayRef containing two positive numbers.
  // * Ceil mode is not supported and must be disabled.
  // * output_max must be greater than output_min.
  //   Namely, setting both output_min and output_max to 0 is not valid usage.
  // * Finally, application of this operator to the input tensor with the given
  //   max pool 2d parameters must result in an output tensor with a valid shape.
  const int64_t pt_outputHeight = pooling_output_shape(
      input.size(Layout::Activation4D::height),
      parameters.kernel[Layout::Parameter::height],
      parameters.padding[Layout::Parameter::height],
      parameters.stride[Layout::Parameter::height],
      parameters.dilation[Layout::Parameter::height],
      ceil_mode);
  const int64_t pt_outputWidth = pooling_output_shape(
      input.size(Layout::Activation4D::width),
      parameters.kernel[Layout::Parameter::width],
      parameters.padding[Layout::Parameter::width],
      parameters.stride[Layout::Parameter::width],
      parameters.dilation[Layout::Parameter::width],
      ceil_mode);
  const int64_t xnnpack_outputHeight = pooling_output_shape(
      input.size(Layout::Activation4D::height),
      parameters.kernel[Layout::Parameter::height],
      parameters.padding[Layout::Parameter::height],
      parameters.stride[Layout::Parameter::height],
      parameters.dilation[Layout::Parameter::height],
      false);
  const int64_t xnnpack_outputWidth = pooling_output_shape(
      input.size(Layout::Activation4D::width),
      parameters.kernel[Layout::Parameter::width],
      parameters.padding[Layout::Parameter::width],
      parameters.stride[Layout::Parameter::width],
      parameters.dilation[Layout::Parameter::width],
      false);

  const bool output_size_eq = (pt_outputHeight == xnnpack_outputHeight) &&
    (pt_outputWidth == xnnpack_outputWidth);

  return xnnpack::available() &&
      // Input
      (4 == input.dim()) &&
      (input.device().is_cpu()) &&
      (kFloat == input.scalar_type()) &&
      !input.requires_grad() &&
      // Kernel
      (2 == parameters.kernel.size()) &&
      (parameters.kernel[Layout::Parameter::height] > 0) &&
      (parameters.kernel[Layout::Parameter::width] > 0) &&
      ((parameters.kernel[Layout::Parameter::height] *
        parameters.kernel[Layout::Parameter::width]) > 1) &&
      // Padding
      (2 == parameters.padding.size()) &&
      (parameters.padding[Layout::Parameter::height] >= 0) &&
      (parameters.padding[Layout::Parameter::width] >= 0) &&
      // Stride
      (2 == parameters.stride.size()) &&
      (parameters.stride[Layout::Parameter::height] > 0) &&
      (parameters.stride[Layout::Parameter::width] > 0) &&
      // Dilation
      (2 == parameters.dilation.size()) &&
      (parameters.dilation[Layout::Parameter::height] > 0) &&
      (parameters.dilation[Layout::Parameter::width] > 0) &&
      // Ceil Mode
      (!ceil_mode || output_size_eq) &&
      // Output Min / Max
      (output_max > output_min) &&
      // Output
      (pooling_output_shape(
        input.size(Layout::Activation4D::height),
        parameters.kernel[Layout::Parameter::height],
        parameters.padding[Layout::Parameter::height],
        parameters.stride[Layout::Parameter::height],
        parameters.dilation[Layout::Parameter::height],
        ceil_mode) > 0) &&
      (pooling_output_shape(
        input.size(Layout::Activation4D::width),
        parameters.kernel[Layout::Parameter::width],
        parameters.padding[Layout::Parameter::width],
        parameters.stride[Layout::Parameter::width],
        parameters.dilation[Layout::Parameter::width],
        ceil_mode) > 0) &&
      true;
}

Tensor max_pool2d(
    const Tensor& input,
    const IntArrayRef kernel_,
    const IntArrayRef padding_,
    IntArrayRef stride_,
    const IntArrayRef dilation_,
    const bool ceil_mode,
    const float output_min,
    const float output_max) {
  using namespace internal;

  // A call to max_pool2d must have been gated by a call to use_maxpool2d, so
  // the parameters are guaranteed to be valid at this point.  Still, stride can
  // be empty, and the parameters not normalized.

  if (stride_.empty()) {
    stride_ = kernel_;
  }

  const internal::pooling::Parameters parameters{
    kernel_,
    padding_,
    stride_,
    dilation_,
  };

  const Tensor input_padded_contig_nhwc =
      mobile::allocate_padded_contiguous_if_needed(
          input,
          MemoryFormat::ChannelsLast);

  Tensor output_padded_contig_nhwc = mobile::empty_with_tail_padding(
      {
        input_padded_contig_nhwc.size(Layout::Activation4D::batch),
        input_padded_contig_nhwc.size(Layout::Activation4D::channels),
        pooling_output_shape(
            input_padded_contig_nhwc.size(Layout::Activation4D::height),
            parameters.kernel[Layout::Parameter::height],
            parameters.padding[Layout::Parameter::height],
            parameters.stride[Layout::Parameter::height],
            parameters.dilation[Layout::Parameter::height],
            ceil_mode),
        pooling_output_shape(
            input_padded_contig_nhwc.size(Layout::Activation4D::width),
            parameters.kernel[Layout::Parameter::width],
            parameters.padding[Layout::Parameter::width],
            parameters.stride[Layout::Parameter::width],
            parameters.dilation[Layout::Parameter::width],
            ceil_mode),
      },
      input_padded_contig_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      input_padded_contig_nhwc.opt_names());

  xnn_operator_t max_pool_op{};

  const xnn_status create_status = xnn_create_max_pooling2d_nhwc_f32(
      parameters.padding[Layout::Parameter::height],                  // input_padding_top
      parameters.padding[Layout::Parameter::width],                   // input_padding_right
      parameters.padding[Layout::Parameter::height],                  // input_padding_bottom
      parameters.padding[Layout::Parameter::width],                   // input_padding_left
      parameters.kernel[Layout::Parameter::height],                   // kernel_height
      parameters.kernel[Layout::Parameter::width],                    // kernel_width
      parameters.stride[Layout::Parameter::height],                   // subsampling_height
      parameters.stride[Layout::Parameter::width],                    // subsampling_width
      parameters.dilation[Layout::Parameter::height],                 // dilation_height
      parameters.dilation[Layout::Parameter::width],                  // dilation_width
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      &max_pool_op);                                                  // operator

  Operator max_pool_scoped_op(max_pool_op);

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_max_pooling2d_nhwc_f32 failed!");

  const xnn_status reshape_status = xnn_reshape_max_pooling2d_nhwc_f32(
      max_pool_op,                                                    // operator
      input_padded_contig_nhwc.size(Layout::Activation4D::batch),     // batch_size
      input_padded_contig_nhwc.size(Layout::Activation4D::height),    // input_height
      input_padded_contig_nhwc.size(Layout::Activation4D::width),     // input_width
      input_padded_contig_nhwc.size(Layout::Activation4D::channels),  // channels
      input_padded_contig_nhwc.size(Layout::Activation4D::channels),  // input_pixel_stride - NHWC Contiguous
      output_padded_contig_nhwc.size(Layout::Activation4D::channels), // output_pixel_stride - NHWC Contiguous
      nullptr,                                                        // output_height_out
      nullptr,                                                        // output_width_out
      caffe2::pthreadpool_());                                        // threadpool

  TORCH_CHECK(
    xnn_status_success == reshape_status,
    "xnn_reshape_max_pooling2d_nhwc_f32 failed!");

  const xnn_status setup_status = xnn_setup_max_pooling2d_nhwc_f32(
      max_pool_op,                                                  // operator
      input_padded_contig_nhwc.data_ptr<float>(),                   // input
      output_padded_contig_nhwc.data_ptr<float>());                 // output

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_max_pooling2d_nhwc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      max_pool_op,              // operator
      caffe2::pthreadpool_());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output_padded_contig_nhwc.contiguous(input.suggest_memory_format());
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `internal`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/Pool.h`
- `ATen/native/utils/Factory.h`
- `ATen/native/xnnpack/Common.h`
- `ATen/native/xnnpack/Engine.h`
- `ATen/native/xnnpack/Pooling.h`


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

Files in the same folder (`aten/src/ATen/native/xnnpack`):

- [`Engine.h_docs.md`](./Engine.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`ChannelShuffle.cpp_docs.md`](./ChannelShuffle.cpp_docs.md)
- [`Convolution.h_docs.md`](./Convolution.h_docs.md)
- [`RegisterOpContextClass.cpp_docs.md`](./RegisterOpContextClass.cpp_docs.md)
- [`Common.h_docs.md`](./Common.h_docs.md)
- [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`Linear.h_docs.md`](./Linear.h_docs.md)
- [`Shim.cpp_docs.md`](./Shim.cpp_docs.md)


## Cross-References

- **File Documentation**: `MaxPooling.cpp_docs.md`
- **Keyword Index**: `MaxPooling.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
