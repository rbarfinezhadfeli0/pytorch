# Documentation: `docs/aten/src/ATen/native/quantized/cpu/QnnpackUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/QnnpackUtils.h_docs.md`
- **Size**: 20,187 bytes (19.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/QnnpackUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/QnnpackUtils.h`
- **Size**: 17,409 bytes (17.00 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#ifdef USE_PYTORCH_QNNPACK
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/utils/Factory.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <utility>
inline int kPaddingChannels = 8;
struct QnnpackOperatorDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);
  }
};

// PackedWeight struct for QNNPACK stores the original Weight and Bias as
// QNNPACK currently does not support an unpack function.
// For PyTorch Mobile, once the model is scripted and serialized we don't need
// to call unpack, so we can save some memory by checking for this case and free
// the original weights after packing.
// Input scale is set to null in pre-pack step. QNNPACK needs bias quantized
// with input scale which is available at runtime in pytorch. During runtime if
// input scale value changes then we requantize bias with the updated scale. For
// inference we expect the graph to be static so the input scale should not
// change across consecutive inference calls.
struct PackedLinearWeightsQnnp : public LinearPackedParamsBase {
  PackedLinearWeightsQnnp(
      std::unique_ptr<qnnpack::PackBMatrix> w,
      at::Tensor orig_weight,
      at::Tensor bias,
      std::optional<double> input_scale,
      at::Tensor w_scales,
      std::vector<uint8_t>&& w_zps)
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias_(at::native::mobile::allocate_padded_contiguous_if_needed(
            bias, bias.suggest_memory_format())),
        per_channel_(this->orig_weight.qscheme() == at::kPerChannelAffine),
        input_scale(std::move(input_scale)),
        w_scales(std::move(w_scales)),
        w_zero_points(std::move(w_zps)),
        q_scheme(this->orig_weight.qscheme()) {
    weight_sizes = this->orig_weight.sizes().vec();
  }

  std::unique_ptr<qnnpack::PackBMatrix> w;
  at::Tensor orig_weight;
  at::Tensor bias_;
  bool per_channel_;
  std::optional<double> input_scale;
  at::Tensor w_scales;
  std::vector<uint8_t> w_zero_points;
  std::vector<float> requantization_scales;
  std::vector<int64_t> weight_sizes;
  c10::QScheme q_scheme;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range=false) override;
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range=false) override;

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  std::optional<at::Tensor> bias() override {
    return bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);

  bool per_channel() const {
    return per_channel_;
  }

 private:
  std::mutex qnnp_mutex_;

#ifdef USE_XNNPACK
  xnnpack_operator xnnp_linear_op;

  template <typename scalar_t, bool kReluFused>
  at::Tensor apply_impl_xnnp(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
#endif // USE_XNNPACK

  template <bool ReluFused>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range);
};

template <int kSpatialDim = 2>
struct PackedConvWeightsQnnp : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightsQnnp(
      std::unique_ptr<qnnpack::PrePackConvWeights> w,
      at::Tensor orig_weight,
      at::Tensor bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose,
      std::optional<double> input_scale,
      std::vector<int64_t> kernel,
      at::Tensor w_scale,
      std::vector<uint8_t>&& w_zps,
      bool is_per_channel)
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose),
        is_per_channel_(is_per_channel),
        input_scale(input_scale),
        kernel_(std::move(kernel)),
        w_scales(std::move(w_scale)),
        w_zero_points(std::move(w_zps)) {
    const bool any_padding = std::any_of(
        padding_.begin(), padding_.end(), [](const auto& e) { return e != 0; });
    const size_t kernel_size =
        std::accumulate(kernel_.begin(), kernel_.end(), 1, std::multiplies<>());

    const size_t group_input_channels = transpose
        ? this->orig_weight.size(0) / groups
        : this->orig_weight.size(1);
    const size_t group_output_channels = transpose
        ? this->orig_weight.size(1)
        : this->orig_weight.size(0) / groups;

    const size_t kernel_depth = kSpatialDim == 3 ? kernel_[0] : 1;
    const size_t kernel_height = kernel_[kSpatialDim - 2];
    const size_t kernel_width = kernel_[kSpatialDim - 1];

    pytorch_qnnp_ukernel_type ukernel_type;
    if (transpose_) {
      ukernel_type = pytorch_qnnp_ukernel_type_conv;
    } else {
      ukernel_type = pytorch_qnnp_ukernel_type_none;

      const bool has_depthwise_dimensions =
          (kSpatialDim == 2 &&
           ((kernel_height == 3 && kernel_width == 3) ||
            (kernel_height == 5 && kernel_width == 5))) ||
          (kSpatialDim == 3 && kernel_height == 3 && kernel_width == 3 &&
           kernel_depth == 3);
      const bool has_depthwise_grouping =
          group_input_channels == 1 && group_output_channels == 1 && groups > 1;

      if (has_depthwise_dimensions && has_depthwise_grouping) {
        ukernel_type = pytorch_qnnp_ukernel_type_dwconv;
      } else if (
          kernel_size == 1 &&
          std::all_of(
              stride_.begin(),
              stride_.end(),
              [](const auto& e) { return e == 1; }) &&
          !any_padding) {
        ukernel_type = group_input_channels >= SIZE_MAX
            ? pytorch_qnnp_ukernel_type_xzp_gemm
            : pytorch_qnnp_ukernel_type_gemm;
      } else {
        ukernel_type = pytorch_qnnp_ukernel_type_conv;
      }
    }

    if (is_per_channel && ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
      TORCH_INTERNAL_ASSERT(
          false, "Per channel quantized weights are not supported for XZP kernels");
    }

    pytorch_qnnp_operator_t convolution{nullptr};
    // Initially all the params are set to zero.
    convolution = static_cast<pytorch_qnnp_operator_t>(
        calloc(1, sizeof(struct pytorch_qnnp_operator)));
    if (convolution == nullptr) {
      TORCH_INTERNAL_ASSERT(
          false, "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
          sizeof(struct pytorch_qnnp_operator));
    }

    convolution_op =
        std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>(
            convolution);

    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    convolution->ukernel_type = ukernel_type;
    convolution->groups = groups;
    convolution->group_input_channels = group_input_channels;
    convolution->group_output_channels = group_output_channels;
    convolution->kernel_depth = kernel_depth;
    convolution->kernel_height = kernel_height;
    convolution->kernel_width = kernel_width;
    convolution->stride_depth = kSpatialDim == 3 ? stride_[0] : 1;
    convolution->stride_height = stride_[kSpatialDim - 2];
    convolution->stride_width = stride_[kSpatialDim - 1];
    convolution->dilation_depth = kSpatialDim == 3 ? dilation_[0] : 1;
    convolution->dilation_height = dilation_[kSpatialDim - 2];
    convolution->dilation_width = dilation_[kSpatialDim - 1];
    convolution->input_padding_height = padding_[kSpatialDim - 2];
    convolution->input_padding_width = padding_[kSpatialDim - 1];
    convolution->input_padding_depth = kSpatialDim == 3 ? padding_[0] : 0;
    convolution->per_channel = is_per_channel_;
    convolution->transpose = transpose_;

    const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
    const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;

    size_t zero_size = sizeof(uint8_t) * k_stride;
    size_t zero_offset = 0;

    if (transpose_) {
      convolution->adjustment_width = output_padding_[1];
      convolution->adjustment_height = output_padding_[0];
      if (group_input_channels < 8) {
        zero_size += 8;
        zero_offset = 8;
      }
    } else {
      zero_buffer_size = 0;
      if (any_padding) {
        zero_size = 0;
        zero_offset = 0;
        if (ukernel_type == pytorch_qnnp_ukernel_type_dwconv) {
          const uint32_t cr = pytorch_qnnp_params.q8dw9.cr;
          const size_t group_stride = (groups + (cr - 1)) & -cr;
          if (groups >= 8) {
            zero_size = sizeof(uint8_t) * group_stride;
            zero_offset = 0;
          } else {
            zero_size = sizeof(uint8_t) * group_stride + 8;
            zero_offset = sizeof(uint8_t) * 8;
          }
        } else if (
            ukernel_type == pytorch_qnnp_ukernel_type_conv ||
            ukernel_type == pytorch_qnnp_ukernel_type_gemm) {
          if (group_input_channels >= 8) {
            zero_size = sizeof(uint8_t) * k_stride;
            zero_offset = 0;
          } else {
            zero_size = sizeof(uint8_t) * k_stride + 8;
            zero_offset = 8;
          }
        }
      }
    }

    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    void* zero_buffer = malloc(zero_size);
    if (zero_buffer == nullptr) {
      pytorch_qnnp_delete_operator(convolution);
      TORCH_INTERNAL_ASSERT(
          false, "failed to allocate %zu bytes for zero padding",
          zero_size);
    }
    // Need to set to input zero point
    // memset(zero_buffer, input_zero_point, zero_size);
    zero_buffer_size = zero_size;
    convolution->zero_buffer = zero_buffer;
    convolution->zero_pointer = (void*)((uintptr_t)zero_buffer + zero_offset);
  }

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter> convolution_op;
  #ifdef USE_XNNPACK
  xnnpack_operator xnnp_convolution_op;
  #endif  // USE_XNNPACK
  std::unique_ptr<qnnpack::PrePackConvWeights> w;
  at::Tensor orig_weight;
  at::Tensor bias;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  bool transpose_;
  bool is_per_channel_;
  std::optional<double> input_scale;
  std::vector<int64_t> kernel_;
  at::Tensor w_scales;
  std::vector<uint8_t> w_zero_points;
  std::vector<float> requantization_scales;
  size_t zero_buffer_size;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(
      const at::Tensor& input,
      bool reduce_range=false) override;

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  int64_t groups() const override {
    return groups_;
  }

  bool transpose() const override {
    return transpose_;
  }

  bool per_channel() const {
    return is_per_channel_;
  }

 private:
  std::mutex qnnp_mutex_;
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

#ifdef USE_XNNPACK
  template <typename scalar_t, bool ReluFused>
  at::Tensor apply_impl_xnnp(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
#endif // USE_XNNPACK
};

enum class Activation : uint8_t { NONE = 0, RELU = 1 };

template<typename T>
inline T QuantizeValue(float scale, int32_t zero_point, float value) {
  const int32_t qmin = std::numeric_limits<T>::min();
  const int32_t qmax = std::numeric_limits<T>::max();
  auto r = zero_point + static_cast<int32_t>(std::nearbyint(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<T>(r);
}

template<typename T>
inline std::pair<T, T> activationLimits(
    float scale,
    int32_t zero_point,
    Activation Ac) {
  switch (Ac) {
    case Activation::NONE:
      return {std::numeric_limits<T>::min(),
              std::numeric_limits<T>::max()};
    case Activation::RELU:
      return {QuantizeValue<T>(scale, zero_point, 0.0),
              std::numeric_limits<T>::max()};
    default:
#ifdef _MSC_VER
      __assume(0);
#else
      __builtin_unreachable();
#endif
  }
}

namespace at::native::qnnp_avgpool_helper {
Tensor qnnpack_avg_pool2d(
    Tensor input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override);
} // namespace at::native::qnnp_avgpool_helper

namespace {
[[maybe_unused]] std::vector<float> generate_requantization_scales(
    const at::Tensor& weight_scales,
    const float input_scale,
    const float output_scale,
    std::vector<float>& requant_scales) {
  // Since weight scale is allocated with padding
  // weight_scales.numel() gives us padded num elements.
  const auto num_output_channels_padded = weight_scales.numel();
  float *const weight_scales_data = weight_scales.data_ptr<float>();
  if (static_cast<int64_t>(requant_scales.size()) < num_output_channels_padded) {
    requant_scales.resize(num_output_channels_padded);
  }
  for (const auto i : c10::irange(num_output_channels_padded)) {
    const auto inverse_output_scale = 1.f /output_scale;
    requant_scales[i] = (weight_scales_data[i] * input_scale) * inverse_output_scale;
    TORCH_CHECK(
        (requant_scales[i] > 0.0f && std::isnormal(requant_scales[i])),
        "failed to create op with requantization scale: ",
        requant_scales[i],
        ": requantization scale must be finite and positive");
  }
  return requant_scales;
}

[[maybe_unused]] std::pair<std::vector<uint8_t>, at::Tensor>
make_zero_points_and_scales_tensor(
    const at::Tensor& weight_contig,
    bool transpose = false,
    uint32_t groups = 1) {
  const int out_ch_idx = transpose ? 1 : 0;
  const auto num_output_channels = weight_contig.size(out_ch_idx) * (transpose ? groups : 1);
  // Add 8 to account for buffering needed by QNNPACK.
  const auto num_output_channels_padded = num_output_channels + kPaddingChannels;
  const auto qtype = weight_contig.qscheme();
  std::vector<uint8_t> weight_zp(num_output_channels_padded, 0);
  // Adjust weight zero point, similar to weight data.
  if (qtype == at::kPerTensorAffine) {
    for (const auto i : c10::irange(num_output_channels)) {
      weight_zp[i] = (uint8_t)(weight_contig.q_zero_point() + 128);
    }
  } else if (qtype == at::kPerChannelAffine) {
    TORCH_CHECK(
        weight_contig.q_per_channel_zero_points().scalar_type() == at::kLong,
        "Per channel zero points dtype must be long int.");
    const int64_t* per_channel_zero_points =
      weight_contig.q_per_channel_zero_points().data_ptr<int64_t>();
    for (const auto i : c10::irange(num_output_channels)) {
      weight_zp[i] = (uint8_t)(per_channel_zero_points[i] + 128);
    }
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported quantization scheme.");
  }
  at:: Tensor weight_scales =
    at::empty(
        {num_output_channels_padded},
        at::device(at::kCPU).dtype(at::kFloat));
  float *const weight_scales_data = weight_scales.data_ptr<float>();
  if (qtype == at::kPerTensorAffine) {
    for (const auto i : c10::irange(num_output_channels)) {
      weight_scales_data[i] = weight_contig.q_scale();
    }
  } else if (qtype == at::kPerChannelAffine) {
    TORCH_CHECK(
        weight_contig.q_per_channel_scales().scalar_type() == at::kDouble,
        "Per channel scales dtype must be double.");
    const double *const per_channel_scales =
      weight_contig.q_per_channel_scales().data_ptr<double>();
    for (const auto i : c10::irange(num_output_channels)) {
      weight_scales_data[i] = static_cast<float>(per_channel_scales[i]);
    }
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported quantization scheme.");
  }
  for (const auto i : c10::irange(num_output_channels, num_output_channels_padded)) {
    weight_scales_data[i] = 1.f;
  }
  return {weight_zp, weight_scales};
}
} // namespace

#endif

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 25 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `QnnpackOperatorDeleter`, `for`, `PackedLinearWeightsQnnp`, `PackedConvWeightsQnnp`, `pytorch_qnnp_operator`, `pytorch_qnnp_operator`, `Activation`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `c10/util/irange.h`
- `pytorch_qnnpack.h`
- `qnnpack_func.h`
- `ATen/native/quantized/cpu/XnnpackUtils.h`
- `ATen/native/quantized/PackedParams.h`
- `ATen/native/utils/Factory.h`
- `ATen/Functions.h`
- `ATen/ops/empty.h`
- `utility`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `QnnpackUtils.h_docs.md`
- **Keyword Index**: `QnnpackUtils.h_kw.md`
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

- **File Documentation**: `QnnpackUtils.h_docs.md_docs.md`
- **Keyword Index**: `QnnpackUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
