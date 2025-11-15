# Documentation: `docs/aten/src/ATen/native/xnnpack/Linear.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/xnnpack/Linear.cpp_docs.md`
- **Size**: 9,848 bytes (9.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/xnnpack/Linear.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/xnnpack/Linear.cpp`
- **Size**: 7,364 bytes (7.19 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/xnnpack/Linear.h>

namespace at::native::xnnpack {
namespace internal::linear {

namespace {

// Supports NHWC and NCHW FP32 linear operators.

// TODO: Decouple and improve error handling and messages.
bool available(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const float output_min,
    const float output_max) {
         // XNNPACK
  return xnnpack::available() &&
          // Weight
          (2 == weight.ndimension()) &&
          (weight.device().is_cpu()) &&
          (kFloat == weight.scalar_type()) &&
          !weight.requires_grad() &&
          // Bias
          ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                       (bias->device().is_cpu()) &&
                                       (kFloat == bias->scalar_type()) &&
                                       (weight.size(Layout::Filter::output)) == bias->size(0) &&
                                       !bias->requires_grad())
                                     : true) &&
          // Output Min / Max
          (output_max > output_min) &&
          true;
}

// TODO: Decouple and improve error handling and messages.
bool usable(const Tensor& input) {
         // Input
  return (1 <= input.ndimension()) &&
         (input.device().is_cpu()) &&
         (kFloat == input.scalar_type()) &&
         !input.requires_grad() &&
         true;
}

Tensor create_and_run(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const float output_min,
    const float output_max) {
  return run(
      create(
          weight,
          bias,
          output_min,
          output_max),
      input);
}

} // anonymous namespace

ContextLinear create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const float output_min,
    const float output_max) {
  const Tensor weight_contig = weight.contiguous();

  TORCH_CHECK(
        available(
          weight_contig,
          bias,
          output_min,
          output_max),
      "XNNPACK Linear not available! "
      "Reason: The provided (weight, bias, output_min, output_max) parameters are "
      "either invalid individually or their combination is not supported by XNNPACK.");

  xnn_operator_t linear_op{};

  const xnn_status create_status = xnn_create_fully_connected_nc_f32(
      weight_contig.size(Layout::Filter::input),                        // input_channels
      weight_contig.size(Layout::Filter::output),                       // output_channels
      weight_contig.size(Layout::Filter::input),                        // input_pixel_stride
      weight_contig.size(Layout::Filter::output),                       // output_pixel_stride
      weight_contig.data_ptr<float>(),                                  // kernel
      (bias && bias->defined()) ?
          bias->contiguous().data_ptr<float>() :
          nullptr,                                                      // bias
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      nullptr,                                                        // xnn_caches_t
      nullptr,                                                        // xnn_weights_cache_t
      &linear_op);                                                    // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_fully_connected_nc_f32 failed!");

  return ContextLinear(
    Operator(linear_op),
    weight_contig.size(Layout::Filter::output)
  );
}

Tensor run(
    const ContextLinear& context,
    const Tensor& input) {
  using namespace internal;

  // For compatibility with aten::linear
  auto ip = input;
  if (input.ndimension() == 1) {
    ip = input.unsqueeze(0);
  }

  const Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
      ip, ip.suggest_memory_format());

  TORCH_CHECK(
      usable(padded_input),
      "XNNPACK Linear not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  const IntArrayRef input_size = padded_input.sizes();
  std::vector<int64_t> output_size(input_size.cbegin(), input_size.cend());
  // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
  output_size.back() = context.output_channels;

  Tensor output = mobile::empty_with_tail_padding(
      output_size,
      padded_input.options().dtype(),
      padded_input.suggest_memory_format(),
      padded_input.opt_names());

  const xnn_status reshape_status = xnn_reshape_fully_connected_nc_f32(
      context.op.get(),                                   // operator
      Layout::ActivationND::batch(padded_input.sizes()),  // Batch,
      caffe2::pthreadpool_());                            // threadpool

  TORCH_CHECK(
      xnn_status_success == reshape_status,
      "xnn_reshape_fully_connected_nc_f32 failed!");

  const xnn_status setup_status = xnn_setup_fully_connected_nc_f32(
      context.op.get(),                                   // operator
      padded_input.data_ptr<float>(),                     // input
      output.data_ptr<float>());                          // output

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_fully_connected_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.op.get(),         // operator
      caffe2::pthreadpool_());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  // For compatibility with aten::linear
  if (input.ndimension() == 1) {
      output.squeeze_(0);
  }

  return output;
}

c10::intrusive_ptr<xnnpack::LinearOpContext> createLinearClampPrePackOpContext(
    Tensor weight,
    std::optional<Tensor> bias,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return xnnpack::XNNPackLinearOpContext::create_context(
      std::move(weight), std::move(bias), output_min, output_max);
}

Tensor linear_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::LinearOpContext>& op_context) {
  return op_context->run(input);
}

IValue
unpack_prepacked_sizes_linear(const IValue& ivalue) {
  auto op_context = ivalue.toCustomClass<xnnpack::LinearOpContext>();
  const auto tuple = op_context->unpack();
  const auto& bias = std::get<1>(tuple);
  return IValue(std::make_tuple(
      std::get<0>(tuple).sizes(),
      (bias && bias->defined()) ? at::OptionalIntArrayRef(bias->sizes()) : std::nullopt));
}

} // namespace internal::linear

bool use_linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  return internal::linear::available(
            weight,
            bias,
            ContextLinear::kMin,
            ContextLinear::kMax) &&
         internal::linear::usable(input);
}

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  return internal::linear::create_and_run(
      input,
      weight,
      bias,
      ContextLinear::kMin,
      ContextLinear::kMax);
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ContextLinear`, `internal`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/xnnpack/Common.h`
- `ATen/native/utils/Factory.h`
- `ATen/native/xnnpack/Linear.h`


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

Files in the same folder (`aten/src/ATen/native/xnnpack`):

- [`Engine.h_docs.md`](./Engine.h_docs.md)
- [`ChannelShuffle.cpp_docs.md`](./ChannelShuffle.cpp_docs.md)
- [`Convolution.h_docs.md`](./Convolution.h_docs.md)
- [`RegisterOpContextClass.cpp_docs.md`](./RegisterOpContextClass.cpp_docs.md)
- [`Common.h_docs.md`](./Common.h_docs.md)
- [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`Linear.h_docs.md`](./Linear.h_docs.md)
- [`Shim.cpp_docs.md`](./Shim.cpp_docs.md)


## Cross-References

- **File Documentation**: `Linear.cpp_docs.md`
- **Keyword Index**: `Linear.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/xnnpack`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/xnnpack`):

- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`Convolution.cpp_docs.md_docs.md`](./Convolution.cpp_docs.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`Pooling.h_docs.md_docs.md`](./Pooling.h_docs.md_docs.md)
- [`RegisterOpContextClass.cpp_kw.md_docs.md`](./RegisterOpContextClass.cpp_kw.md_docs.md)
- [`AveragePooling.cpp_kw.md_docs.md`](./AveragePooling.cpp_kw.md_docs.md)
- [`OpContext.cpp_kw.md_docs.md`](./OpContext.cpp_kw.md_docs.md)
- [`ChannelShuffle.cpp_docs.md_docs.md`](./ChannelShuffle.cpp_docs.md_docs.md)
- [`MaxPooling.cpp_kw.md_docs.md`](./MaxPooling.cpp_kw.md_docs.md)
- [`Common.h_docs.md_docs.md`](./Common.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Linear.cpp_docs.md_docs.md`
- **Keyword Index**: `Linear.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
