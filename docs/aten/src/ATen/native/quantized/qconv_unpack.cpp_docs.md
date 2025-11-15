# Documentation: `aten/src/ATen/native/quantized/qconv_unpack.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/qconv_unpack.cpp`
- **Size**: 9,458 bytes (9.24 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
/*
The dispatch registrations at the end of this file applies to fbgemm, qnnpack, and cudnn backends.
The correct unpack backend function is determined using runtime polymorphism through the packed_weight pointer,
which is of type intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> and points to either a PackedConvWeightsQnnp,
PackedConvWeights (Fbgemm), or PackedConvWeightsCudnn at runtime, which all inherit from ConvPackedParamsBase.
The implementations for the unpack functions can be found in /cpu/qconv_unpack_impl.cpp, for fbgemm&qnnpack
and /cudnn/ConvUnpackImpl.cpp, for cudnn.
*/

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <tuple>

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/from_blob.h>
#endif


namespace at::native {
namespace {

/*
 * QConvPackWeightInt8 expects its input tensor to be in shape
 * [output_channels, kernel_height, kernel_width, input_channels/Groups]
 * Therefore, the unpacking of packed weight tensor using QConvUnpackWeightsInt8
 * results in a tensor of the same shape.
 */

template <int kSpatialDim = 2>
class QConvUnpackWeightsInt8 final {
 public:
  static std::tuple<at::Tensor, std::optional<at::Tensor>> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM ||
        ctx.qEngine() == at::QEngine::X86) {
      return packed_weight->unpack();
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          kSpatialDim == 2,
          "quantized::conv2d_unpack (qnnpack): QNNPACK only supports Conv2d "
          "now.");
      return packed_weight->unpack();
    }
#endif

#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return packed_weight->unpack();
    }
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv2d_unpack ",
        toString(ctx.qEngine()));
  }
};

class QConv1dUnpackWeightsInt8 final {
 public:
  static std::tuple<at::Tensor, std::optional<at::Tensor>> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight) {
    auto& ctx = at::globalContext();
    at::Tensor weight;
    std::optional<at::Tensor> bias;
#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM ||
        ctx.qEngine() == at::QEngine::X86) {
      std::tie(weight, bias) = packed_weight->unpack();
      weight = weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      return std::tuple<at::Tensor, std::optional<at::Tensor>>(weight, bias);
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      std::tie(weight, bias) = packed_weight->unpack();
      at::Tensor new_weight = weight.clone();
      new_weight = new_weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      return std::tuple<at::Tensor, std::optional<at::Tensor>>(new_weight, bias);
    }
#endif

#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      std::tie(weight, bias) = packed_weight->unpack();
      at::Tensor new_weight = weight.clone();
      new_weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      return std::tuple<at::Tensor, std::optional<at::Tensor>>(new_weight, bias);
    }
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv1d_unpack ",
        toString(ctx.qEngine()));
  }
};

template <int kSpatialDim = 2>
class QConvStride final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->stride();
  }
};

template <int kSpatialDim = 2>
class QConvPadding final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->padding();
  }
};

template <int kSpatialDim = 2>
class QConvOutputPadding final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->output_padding();
  }
};

template <int kSpatialDim = 2>
class QConvDilation final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->dilation();
  }
};

template <int kSpatialDim = 2>
class QConvGroups final {
 public:
  static int64_t run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->groups();
  }
};

template <int kSpatialDim = 2>
class QConvTranspose final {
 public:
  static int64_t run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->transpose();
  }
};

IValue
unpack_quantized_prepacked_sizes_conv2d(const IValue& ivalue) {
  auto params = ivalue.toCustomClass<ConvPackedParamsBase<2>>();
  auto [weight, bias] = params->unpack();
  at::OptionalIntArrayRef bias_sizes = std::nullopt;
  if (bias && bias->defined()) {
    bias_sizes = bias->sizes();
  }
  return IValue(std::make_tuple(
      weight.sizes(),
      bias_sizes,
      params->stride(),
      params->padding(),
      params->dilation(),
      params->groups()));
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  register_conv_params<2>();
  register_conv_params<3>();
  // conv_unpack is deprecated, please use conv2d_unpack for 2D conv.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
  // We use  conv2d_unpack to be consistent with conv3d_unpack
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_unpack"), TORCH_FN(QConv1dUnpackWeightsInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_unpack_sizes"), TORCH_FN(unpack_quantized_prepacked_sizes_conv2d));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<3>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_stride"), TORCH_FN(QConvStride<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_padding"), TORCH_FN(QConvPadding<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_output_padding"), TORCH_FN(QConvOutputPadding<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_dilation"), TORCH_FN(QConvDilation<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_groups"), TORCH_FN(QConvGroups<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_transpose"), TORCH_FN(QConvTranspose<2>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_stride"), TORCH_FN(QConvStride<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_padding"), TORCH_FN(QConvPadding<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_output_padding"), TORCH_FN(QConvOutputPadding<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_dilation"), TORCH_FN(QConvDilation<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_groups"), TORCH_FN(QConvGroups<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_transpose"), TORCH_FN(QConvTranspose<3>::run));

  // ConvTranspose is the same, however, we want to have different name.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_unpack"), TORCH_FN(QConv1dUnpackWeightsInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<3>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_stride"), TORCH_FN(QConvStride<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_padding"), TORCH_FN(QConvPadding<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_output_padding"), TORCH_FN(QConvOutputPadding<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_dilation"), TORCH_FN(QConvDilation<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_groups"), TORCH_FN(QConvGroups<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_transpose"), TORCH_FN(QConvTranspose<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_stride"), TORCH_FN(QConvStride<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_padding"), TORCH_FN(QConvPadding<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_output_padding"), TORCH_FN(QConvOutputPadding<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_dilation"), TORCH_FN(QConvDilation<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_groups"), TORCH_FN(QConvGroups<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_transpose"), TORCH_FN(QConvTranspose<3>::run));
}

} // namespace
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 8 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `QConvUnpackWeightsInt8`, `QConv1dUnpackWeightsInt8`, `QConvStride`, `QConvPadding`, `QConvOutputPadding`, `QConvDilation`, `QConvGroups`, `QConvTranspose`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `tuple`
- `ATen/core/Tensor.h`
- `ATen/core/List.h`
- `ATen/core/ivalue.h`
- `torch/library.h`
- `ATen/native/quantized/cpu/fbgemm_utils.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `ATen/native/quantized/cpu/OnednnUtils.h`
- `ATen/native/quantized/cpu/QuantUtils.h`
- `ATen/native/quantized/PackedParams.h`
- `ATen/native/quantized/library.h`
- `ATen/Functions.h`
- `ATen/ops/_empty_affine_quantized.h`
- `ATen/ops/_empty_per_channel_affine_quantized.h`
- `ATen/ops/from_blob.h`


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

Files in the same folder (`aten/src/ATen/native/quantized`):

- [`QTensor.cpp_docs.md`](./QTensor.cpp_docs.md)
- [`FakeQuantAffine.h_docs.md`](./FakeQuantAffine.h_docs.md)
- [`AffineQuantizerBase.h_docs.md`](./AffineQuantizerBase.h_docs.md)
- [`AffineQuantizerBase.cpp_docs.md`](./AffineQuantizerBase.cpp_docs.md)
- [`PackedParams.h_docs.md`](./PackedParams.h_docs.md)
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)
- [`AffineQuantizer.h_docs.md`](./AffineQuantizer.h_docs.md)
- [`qlinear_unpack.cpp_docs.md`](./qlinear_unpack.cpp_docs.md)
- [`TensorFactories.cpp_docs.md`](./TensorFactories.cpp_docs.md)


## Cross-References

- **File Documentation**: `qconv_unpack.cpp_docs.md`
- **Keyword Index**: `qconv_unpack.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
