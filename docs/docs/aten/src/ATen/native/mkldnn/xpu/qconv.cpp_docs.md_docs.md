# Documentation: `docs/aten/src/ATen/native/mkldnn/xpu/qconv.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/xpu/qconv.cpp_docs.md`
- **Size**: 12,099 bytes (11.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/xpu/qconv.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/xpu/qconv.cpp`
- **Size**: 9,698 bytes (9.47 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/mkldnn/xpu/qconv.h>

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

using namespace at::native::onednn;
namespace at::native::xpu {

inline c10::ScalarType QConvoneDNNXPU::qconv_decide_out_dtype(
    const at::Tensor& act,
    const std::optional<c10::ScalarType> output_dtype) {
  bool fp32_output = output_dtype.has_value() && (output_dtype == c10::kFloat);
  bool bfloat16_output =
      output_dtype.has_value() && (output_dtype == c10::kBFloat16);
  auto dst_dtype = fp32_output
      ? c10::kFloat
      : (bfloat16_output ? c10::kBFloat16 : act.scalar_type());
  return dst_dtype;
}

at::Tensor QConvoneDNNXPU::qconv_prepack_xpu(
    at::Tensor weight,
    at::Tensor weight_scales,
    double input_scale,
    int64_t input_zero_point,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    std::optional<torch::List<int64_t>> input_shape) {
  // XPU has no prepack at present
  return weight;
}

at::Tensor QConvoneDNNXPU::run_pointwise(
    at::Tensor act,
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    double inv_output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  if (act.dim() == 3 || act.dim() == 5) {
    TORCH_CHECK(
        attr == "none",
        "quantized pointwise conv",
        act.dim() - 2,
        "d doesn't support unary_post_op fusion. Got unary_post_op:",
        attr,
        ".");
  } else {
    TORCH_CHECK(
        attr == "none" || attr == "relu" || attr == "hardtanh" ||
            attr == "hardswish" || attr == "swish",
        "We support quantized convolution without any post-ops or combinations for Quantized Conv + ReLU, Hardtanh, GELU, Swish, and Hardswish are supported. However, encountered unsupported post operation:",
        attr,
        ".");
  }

  bool is_channels_last_suggested = use_channels_last_for_conv(act, weight);
  auto mfmt = is_channels_last_suggested ? get_cl_tag_by_ndim(act.ndimension())
                                         : at::MemoryFormat::Contiguous;
  Tensor input_ = act.contiguous(mfmt);
  Tensor weight_ = weight.contiguous(mfmt);

  auto dst_tz = conv_dst_size(
      input_.ndimension(),
      input_.sizes(),
      weight_.sizes(),
      padding.vec(),
      padding.vec(),
      stride.vec(),
      dilation.vec());

  auto dst_dtype = qconv_decide_out_dtype(act, output_dtype);
  Tensor output =
      at::empty(dst_tz, act.options().dtype(dst_dtype).memory_format(mfmt));

  return quantized_convolution(
      act,
      act_scale,
      act_zero_point,
      weight,
      weight_scales,
      weight_zero_points,
      bias,
      stride,
      padding,
      dilation,
      /*transposed*/ false,
      groups,
      output,
      inv_output_scale,
      output_zero_point,
      /*accum*/ std::nullopt,
      /*accum_scale*/ 0.0,
      /*accum_zero_point*/ 0,
      /*output_dtype*/ output_dtype,
      /*binary_attr*/ std::nullopt,
      /*binary_alpha*/ std::nullopt,
      /*unary_attr*/ attr,
      /*unary_scalars*/ scalars,
      /*unary_algorithm*/ algorithm);
}

at::Tensor QConvoneDNNXPU::run_pointwise_tensor(
    at::Tensor act,
    at::Tensor act_scale,
    at::Tensor act_zero_point,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  return run_pointwise(
      act,
      act_scale.item().toDouble(),
      act_zero_point.item().toLong(),
      weight,
      weight_scales,
      weight_zero_points,
      bias,
      stride,
      padding,
      dilation,
      groups,
      output_scale,
      output_zero_point,
      output_dtype,
      /*unary_attr*/ attr,
      /*unary_scalars*/ scalars,
      /*unary_algorithm*/ algorithm);
}

at::Tensor QConvoneDNNXPU::run_pointwise_binary(
    at::Tensor act,
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    at::Tensor accum,
    std::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    double accum_scale,
    int64_t accum_zero_point,
    std::string_view binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<std::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string_view> unary_algorithm) {
  TORCH_CHECK(
      act.dim() == 4 && binary_attr == "sum" &&
          (!unary_attr.has_value() ||
           (unary_attr.has_value() &&
            (unary_attr.value() == "none" || unary_attr.value() == "relu"))),
      "post_op sum or post_op sum_relu is supported for quantized pointwise conv2d. Got binary_post_op: ",
      binary_attr,
      " unary_post_op: ",
      unary_attr.has_value() ? unary_attr.value() : "none",
      ".")

  bool is_channels_last_suggested = use_channels_last_for_conv(act, weight);
  auto mfmt = is_channels_last_suggested ? get_cl_tag_by_ndim(act.ndimension())
                                         : at::MemoryFormat::Contiguous;
  Tensor input_ = act.contiguous(mfmt);
  Tensor weight_ = weight.contiguous(mfmt);

  auto dst_tz = conv_dst_size(
      input_.ndimension(),
      input_.sizes(),
      weight_.sizes(),
      padding.vec(),
      padding.vec(),
      stride.vec(),
      dilation.vec());

  auto dst_dtype = qconv_decide_out_dtype(act, output_dtype);
  bool has_accum_postop_sum = binary_attr == "sum";
  Tensor output = has_accum_postop_sum
      ? accum
      : at::empty(dst_tz, act.options().dtype(dst_dtype).memory_format(mfmt));

  output = quantized_convolution(
      act,
      act_scale,
      act_zero_point,
      weight,
      weight_scales,
      weight_zero_points,
      bias,
      stride,
      padding,
      dilation,
      /*transposed*/ false,
      groups,
      output,
      output_scale,
      output_zero_point,
      /*accum*/ accum,
      /*accum_scale*/ accum_scale,
      /*accum_zero_point*/ accum_zero_point,
      /*output_dtype*/ output_dtype,
      /*binary_attr*/ binary_attr,
      /*binary_alpha*/ alpha,
      /*unary_attr*/ unary_attr,
      /*unary_scalars*/ unary_scalars,
      /*unary_algorithm*/ unary_algorithm);

  if (!has_accum_postop_sum) {
    return output;
  } else {
    return accum;
  }
}

at::Tensor QConvoneDNNXPU::run_pointwise_binary_tensor(
    at::Tensor act, // contains quantized values but not QTensor
    at::Tensor act_scale,
    at::Tensor act_zero_point,
    at::Tensor weight, // contains quantized values but not QTensor
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    at::Tensor accum, // contains quantized values but not QTensor
    std::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    double accum_scale,
    int64_t accum_zero_point,
    std::string_view binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<std::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string_view> unary_algorithm) {
  return run_pointwise_binary(
      act,
      act_scale.item().toDouble(),
      act_zero_point.item().toLong(),
      weight,
      weight_scales,
      weight_zero_points,
      accum,
      bias,
      stride,
      padding,
      dilation,
      groups,
      output_scale,
      output_zero_point,
      output_dtype,
      accum_scale,
      accum_zero_point,
      binary_attr,
      alpha,
      unary_attr,
      unary_scalars,
      unary_algorithm);
}

TORCH_LIBRARY_IMPL(onednn, XPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qconv_prepack"),
      TORCH_FN(QConvoneDNNXPU::qconv_prepack_xpu));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qconv1d_pointwise"),
      QConvoneDNNXPU::run_pointwise);
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise"),
      QConvoneDNNXPU::run_pointwise);
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qconv3d_pointwise"),
      QConvoneDNNXPU::run_pointwise);
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise.binary"),
      QConvoneDNNXPU::run_pointwise_binary);
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qconv_pointwise"),
      QConvoneDNNXPU::run_pointwise);
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qconv_pointwise.tensor"),
      QConvoneDNNXPU::run_pointwise_tensor);
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise.binary_tensor"),
      QConvoneDNNXPU::run_pointwise_binary_tensor);
}

} // namespace at::native::xpu

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/op_registration/op_registration.h`
- `ATen/native/mkldnn/xpu/detail/oneDNN.h`
- `ATen/native/mkldnn/xpu/qconv.h`
- `c10/core/MemoryFormat.h`
- `c10/core/ScalarType.h`
- `torch/library.h`


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

Files in the same folder (`aten/src/ATen/native/mkldnn/xpu`):

- [`Attention.cpp_docs.md`](./Attention.cpp_docs.md)
- [`Conv.h_docs.md`](./Conv.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`qlinear.h_docs.md`](./qlinear.h_docs.md)
- [`ScaledBlas.cpp_docs.md`](./ScaledBlas.cpp_docs.md)
- [`qconv.h_docs.md`](./qconv.h_docs.md)
- [`FusionUtils.cpp_docs.md`](./FusionUtils.cpp_docs.md)
- [`FusionUtils.h_docs.md`](./FusionUtils.h_docs.md)
- [`Conv.cpp_docs.md`](./Conv.cpp_docs.md)


## Cross-References

- **File Documentation**: `qconv.cpp_docs.md`
- **Keyword Index**: `qconv.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/mkldnn/xpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native/mkldnn/xpu`):

- [`FusionUtils.cpp_kw.md_docs.md`](./FusionUtils.cpp_kw.md_docs.md)
- [`Conv.h_kw.md_docs.md`](./Conv.h_kw.md_docs.md)
- [`qconv.h_docs.md_docs.md`](./qconv.h_docs.md_docs.md)
- [`ScaledBlas.cpp_docs.md_docs.md`](./ScaledBlas.cpp_docs.md_docs.md)
- [`Attention.cpp_kw.md_docs.md`](./Attention.cpp_kw.md_docs.md)
- [`FusionUtils.h_docs.md_docs.md`](./FusionUtils.h_docs.md_docs.md)
- [`Blas.cpp_docs.md_docs.md`](./Blas.cpp_docs.md_docs.md)
- [`Conv.cpp_docs.md_docs.md`](./Conv.cpp_docs.md_docs.md)
- [`qconv.cpp_kw.md_docs.md`](./qconv.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `qconv.cpp_docs.md_docs.md`
- **Keyword Index**: `qconv.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
