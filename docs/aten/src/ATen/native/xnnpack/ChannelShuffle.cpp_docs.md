# Documentation: `aten/src/ATen/native/xnnpack/ChannelShuffle.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/xnnpack/ChannelShuffle.cpp`
- **Size**: 4,468 bytes (4.36 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/utils/Factory.h>

namespace at::native::xnnpack {

bool use_channel_shuffle(
    const Tensor& input,
    const int64_t groups) {
  using namespace internal;

  // Here are the list of conditions required for this code path to be taken:
  // * Input must be 4D CPU float tensor with no gradients and
  //   and all dimensions must be positive.
  // * The number of groups must be larger than 1 and
  //   the number of channels must be divisible by the number of groups.
  return xnnpack::available() &&
      // Input
      (4 == input.dim()) &&
      (input.device().is_cpu()) &&
      (kFloat == input.scalar_type()) &&
      (input.size(Layout::Activation4D::batch) >= 0) &&
      (input.size(Layout::Activation4D::channels) > 0) &&
      (input.size(Layout::Activation4D::height) > 0) &&
      (input.size(Layout::Activation4D::width) > 0) &&
      !input.requires_grad() &&
      // Groups
      groups > 1 &&
      (0 == input.size(Layout::Activation4D::channels) % groups) &&
      true;
}

Tensor channel_shuffle(
    const Tensor& input,
    const int64_t groups) {
  using namespace internal;

  // A call to channel_shuffle must have been gated by a call to use_channel_shuffle,
  // so the parameters are guaranteed to be valid at this point.

  const Tensor input_padded_contig_nhwc =
      mobile::allocate_padded_contiguous_if_needed(
          input,
          MemoryFormat::ChannelsLast);

  Tensor output_padded_contig_nhwc = mobile::empty_with_tail_padding(
      {
        input_padded_contig_nhwc.size(Layout::Activation4D::batch),
        input_padded_contig_nhwc.size(Layout::Activation4D::channels),
        input_padded_contig_nhwc.size(Layout::Activation4D::height),
        input_padded_contig_nhwc.size(Layout::Activation4D::width),
      },
      input_padded_contig_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      input_padded_contig_nhwc.opt_names());

  int64_t channels_per_group =
      input_padded_contig_nhwc.size(Layout::Activation4D::channels) / groups;

  xnn_operator_t channel_shuffle_op{};

  const xnn_status create_status = xnn_create_channel_shuffle_nc_x32(
      groups,                                                         // number of groups
      channels_per_group,                                             // number of channels per group
      input_padded_contig_nhwc.size(Layout::Activation4D::channels),  // input_pixel_stride - NHWC Contiguous
      output_padded_contig_nhwc.size(Layout::Activation4D::channels), // output_pixel_stride - NHWC Contiguous
      0u,                                                             // flags
      &channel_shuffle_op);                                           // operator

  Operator channel_shuffle_scoped_op(channel_shuffle_op);

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_channel_shuffle_nc_x32 failed!");

  int64_t batch_size = input_padded_contig_nhwc.size(Layout::Activation4D::batch) *
                       input_padded_contig_nhwc.size(Layout::Activation4D::height) *
                       input_padded_contig_nhwc.size(Layout::Activation4D::width);

  const xnn_status reshape_status = xnn_reshape_channel_shuffle_nc_x32(
      channel_shuffle_op,                                           // operator
      batch_size,                                                   // batch_size
      caffe2::pthreadpool_());                                      // threadpool

  TORCH_CHECK(
      xnn_status_success == reshape_status,
      "xnn_reshape_channel_shuffle_nc_x32 failed!");

  const xnn_status setup_status = xnn_setup_channel_shuffle_nc_x32(
      channel_shuffle_op,                                           // operator
      input_padded_contig_nhwc.data_ptr<float>(),                   // input
      output_padded_contig_nhwc.data_ptr<float>());                 // output

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_channel_shuffle_nc_x32 failed!");

  const xnn_status run_status = xnn_run_operator(
      channel_shuffle_op,       // operator
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


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

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

- `ATen/native/xnnpack/Common.h`
- `ATen/native/xnnpack/Engine.h`
- `ATen/native/utils/Factory.h`


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
- [`Convolution.h_docs.md`](./Convolution.h_docs.md)
- [`RegisterOpContextClass.cpp_docs.md`](./RegisterOpContextClass.cpp_docs.md)
- [`Common.h_docs.md`](./Common.h_docs.md)
- [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`Linear.h_docs.md`](./Linear.h_docs.md)
- [`Shim.cpp_docs.md`](./Shim.cpp_docs.md)


## Cross-References

- **File Documentation**: `ChannelShuffle.cpp_docs.md`
- **Keyword Index**: `ChannelShuffle.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
