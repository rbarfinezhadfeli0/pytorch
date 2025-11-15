# Documentation: `docs/aten/src/ATen/native/quantized/cpu/Normalization.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/Normalization.cpp_docs.md`
- **Size**: 17,423 bytes (17.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/Normalization.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/Normalization.cpp`
- **Size**: 14,603 bytes (14.26 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/quantized_batch_norm_native.h>
#endif

#include <algorithm>

namespace at::native {

DEFINE_DISPATCH(qbatch_norm_stub);
DEFINE_DISPATCH(qbatch_norm_relu_stub);
DEFINE_DISPATCH(qbatch_norm_cpu_stub);

namespace {
void compute_fused_params(
    const int64_t channels,
    const float* weight_data,
    const float* bias_data,
    const float* mean_data,
    const float* var_data,
    double eps,
    double input_scale,
    double output_scale,
    float* alpha_data,
    float* beta_data) {
  // Batch Normalization
  // output(n, c, h, w)
  //     = (input(n, c, h, w) - mean(c)) / sqrt(var(c) + eps) * weight(c)
  //         + bias(c)
  // We factor out inv_sigma(c) = 1 / sqrt(var(c) + eps).
  for (const auto c : c10::irange(channels)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    float inv_sigma = 1.0 / std::sqrt(var_data[c] + static_cast<float>(eps));
    float weight_v = weight_data ? weight_data[c] : 1;
    float bias_v = bias_data ? bias_data[c] : 0;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    alpha_data[c] = inv_sigma * weight_v * (input_scale / output_scale);
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    beta_data[c] = (bias_v - mean_data[c] * inv_sigma * weight_v) / output_scale;
  }
}

template <bool ReluFused>
Tensor q_batch_norm1d_impl(
    Tensor qx,
    std::optional<Tensor> mb_weight,
    std::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");
  const auto& weight = *mb_weight;
  const auto& bias = *mb_bias;

  if (qx.numel() == 0) {
    auto out = qx.clone();
    return out;
  }
  int64_t ndim = qx.dim();
  TORCH_CHECK(ndim == 2 || ndim == 3, "Expecting the input tensor of rank 2 or 3.");
  const int64_t N = qx.size(0);
  const int64_t C = qx.size(1);
  const int64_t H = ndim == 3 ? qx.size(2) : 1;

  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

  const float* weight_data = weight.template const_data_ptr<float>();
  const float* bias_data = bias.template const_data_ptr<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.mutable_data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  const float* mean_data = mean.template const_data_ptr<float>();
  const float* var_data = var.template const_data_ptr<float>();

  if (ndim == 2) {
    // create a fake H and W dimension so we can use NHWC
    qx = qx.unsqueeze(-1).unsqueeze(-1);
  } else {
    // create a fake W dimension so we can use NHWC
    qx = qx.unsqueeze(-1);
  }

  auto oSizes = qx.sizes();
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
        .dtype(qx_nhwc.scalar_type())
        .memory_format(MemoryFormat::ChannelsLast),
      output_scale,
      output_zero_point,
      std::nullopt);

  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx.q_scale(),
      output_scale,
      alpha_data,
      beta_data);
  if (ReluFused) {
    qbatch_norm_relu_stub(
        qx.device().type(),
        N,
        C,
        H,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  } else {
    qbatch_norm_stub(
        qx.device().type(),
        N,
        C,
        H,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  }
  // Remove the fake dimension, and go back to contiguous format
  // (since there is no 4th channel). Note, this has a performance
  // cost.
  Tensor result = qy.contiguous(MemoryFormat::Contiguous).squeeze(-1);
  if (ndim == 2) {
    result = result.squeeze(-1);
  }
  return result;
}

template <bool ReluFused>
Tensor q_batch_norm2d_impl(
    Tensor qx,
    std::optional<Tensor> mb_weight,
    std::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");
  const auto& weight = *mb_weight;
  const auto& bias = *mb_bias;

  if (qx.numel() == 0) {
    auto out = qx.clone();
    return out;
  }
  int64_t ndim = qx.dim();
  TORCH_CHECK(ndim == 4, "Expecting the input tensor of rank 4.");
  const int64_t N = qx.size(0);
  const int64_t C = qx.size(1);
  const int64_t H = qx.size(2);
  const int64_t W = qx.size(3);

  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

  const float* weight_data = weight.template const_data_ptr<float>();
  const float* bias_data = bias.template const_data_ptr<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.mutable_data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  const float* mean_data = mean.template const_data_ptr<float>();
  const float* var_data = var.template const_data_ptr<float>();

  auto oSizes = qx.sizes();
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
        .dtype(qx_nhwc.scalar_type())
        .memory_format(MemoryFormat::ChannelsLast),
      output_scale,
      output_zero_point,
      std::nullopt);

  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx.q_scale(),
      output_scale,
      alpha_data,
      beta_data);
  if (ReluFused) {
    qbatch_norm_relu_stub(
        qx.device().type(),
        N,
        C,
        H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  } else {
    qbatch_norm_stub(
        qx.device().type(),
        N,
        C,
        H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  }
  return qy;
}

template <bool ReluFused>
Tensor q_batch_norm3d_impl(
    Tensor qx,
    std::optional<Tensor> mb_weight,
    std::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided")
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided")

  const auto& weight = *mb_weight;
  const auto& bias = *mb_bias;

  if (qx.numel() == 0) {
    auto out = qx.clone();
    return out;
  }
  int64_t ndim = qx.dim();
  TORCH_CHECK(ndim == 5, "Expecting the input tensor of rank 5.");
  const int64_t N = qx.size(0);
  const int64_t C = qx.size(1);
  const int64_t D = qx.size(2);
  const int64_t H = qx.size(3);
  const int64_t W = qx.size(4);

  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

  const float* weight_data = weight.template const_data_ptr<float>();
  const float* bias_data = bias.template const_data_ptr<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.mutable_data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  const float* mean_data = mean.template const_data_ptr<float>();
  const float* var_data = var.template const_data_ptr<float>();

  auto oSizes = qx.sizes();
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast3d);
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
        .dtype(qx_nhwc.scalar_type())
        .memory_format(MemoryFormat::ChannelsLast3d),
      output_scale,
      output_zero_point,
      std::nullopt);

  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx.q_scale(),
      output_scale,
      alpha_data,
      beta_data);

  if (ReluFused) {
    qbatch_norm_relu_stub(
        qx.device().type(),
        N,
        C,
        D * H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  } else {
    qbatch_norm_stub(
        qx.device().type(),
        N,
        C,
        D * H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  }
  return qy;
}

template <bool ReluFused>
Tensor q_batch_norm_impl(
    Tensor qx,
    std::optional<Tensor> mb_weight,
    std::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {
  Tensor qy;
  int64_t dim = qx.dim();
  if (dim == 2 || dim == 3) {
    qy = q_batch_norm1d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else if (dim == 4) {
    qy = q_batch_norm2d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else if (dim == 5) {
    qy = q_batch_norm3d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else {
    TORCH_CHECK(false, "quantized::batch_norm only support 2d, 3d, 4d or 5d inputs.");
  }
  return qy;
}

Tensor int8_batch_norm2d_cpu_impl(
    const Tensor& qx,
    double qx_scale,
    int64_t qx_zero_point,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& mean,
    const Tensor& var,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    c10::ScalarType output_dtype) {
  if (qx.numel() == 0) {
    auto out = qx.clone();
    return out;
  }
  if (output_dtype != at::kByte) {
    TORCH_CHECK(output_scale == 1.0 && output_zero_point == 0,
                "Quantized batch_norm_2d output scale and zero point should be 1 and 0 for "
                "output_dtype ", output_dtype, ", but got scale = ",
                output_scale, " and zero point = ", output_zero_point);
  }
  int64_t ndim = qx.dim();
  TORCH_CHECK(ndim == 4, "Int8 batch_norm2d: Expecting the input tensor of rank 4.");
  const int64_t N = qx.size(0);
  const int64_t C = qx.size(1);
  const int64_t H = qx.size(2);
  const int64_t W = qx.size(3);

  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

  const float* weight_data = weight.template const_data_ptr<float>();
  const float* bias_data = bias.template const_data_ptr<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.mutable_data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  const float* mean_data = mean.template const_data_ptr<float>();
  const float* var_data = var.template const_data_ptr<float>();

  auto oSizes = qx.sizes();
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::empty(
      oSizes,
      at::device(kCPU)
        .dtype(output_dtype)
        .memory_format(MemoryFormat::ChannelsLast));

  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx_scale,
      output_scale,
      alpha_data,
      beta_data);
  qbatch_norm_cpu_stub(
      qx.device().type(),
      N,
      C,
      H * W,
      qx_zero_point,
      output_zero_point,
      qx_nhwc,
      alpha,
      beta,
      qy);
  return qy;
}

} // namespace

Tensor quantized_batch_norm(
    const Tensor& qx, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */,
    const Tensor& mean /* optional */,
    const Tensor& var /* optional */,
    double eps,
    double output_scale,
    int64_t output_zero_point) {
  return q_batch_norm_impl<false>(
      qx,
      weight_opt,
      bias_opt,
      mean,
      var,
      eps,
      output_scale,
      output_zero_point);
}


TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm"),        TORCH_FN(q_batch_norm_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm_relu"),   TORCH_FN(q_batch_norm_impl<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm1d"),      TORCH_FN(q_batch_norm1d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm1d_relu"), TORCH_FN(q_batch_norm1d_impl<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm2d"),      TORCH_FN(q_batch_norm2d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm2d_relu"), TORCH_FN(q_batch_norm2d_impl<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm3d"),      TORCH_FN(q_batch_norm3d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm3d_relu"), TORCH_FN(q_batch_norm3d_impl<true>));
}

TORCH_LIBRARY_IMPL(onednn, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("onednn::qbatch_norm2d"), TORCH_FN(int8_batch_norm2d_cpu_impl));
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Parallel.h`
- `torch/library.h`
- `ATen/native/quantized/cpu/QuantizedOps.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_empty_affine_quantized.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/empty.h`
- `ATen/ops/quantized_batch_norm_native.h`
- `algorithm`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `Normalization.cpp_docs.md`
- **Keyword Index**: `Normalization.cpp_kw.md`
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

- **File Documentation**: `Normalization.cpp_docs.md_docs.md`
- **Keyword Index**: `Normalization.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
