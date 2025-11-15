# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qconv_unpack_impl.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qconv_unpack_impl.cpp_docs.md`
- **Size**: 7,893 bytes (7.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qconv_unpack_impl.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qconv_unpack_impl.cpp`
- **Size**: 5,210 bytes (5.09 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/PackedParams.h>

#ifdef USE_FBGEMM
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<
    kSpatialDim>::unpack() {
  auto* packed_weights_p = w.get();
  // output channels
  const int output_channels = packed_weights_p->outputChannels();
  const int input_channels = packed_weights_p->inputChannels();
  const int groups = packed_weights_p->groups();

  const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
  // R (kernel height)
  const int kernel_h = kernel[kSpatialDim - 2];
  // S (kernel width)
  const int kernel_w = kernel[kSpatialDim - 1];

  const int C_per_G = input_channels / groups;

  // Tensor for unpacked weights
  // Unpacked format would be physical KRS(C/G) but logical KCRS (channels
  // first) because that's how
  // ChannelsLast3d is not available now.FBGEMM stores the weights
  // TODO: Unify 2d and 3d when ChannelsLast3d is ready.
  at::Tensor unpacked_weights;
  if (q_scheme == c10::kPerTensorAffine) {
    unpacked_weights = kSpatialDim == 2
        ? at::_empty_affine_quantized(
              {output_channels, C_per_G, kernel_h, kernel_w},
              at::device(c10::kCPU)
                  .dtype(c10::kQInt8)
                  .memory_format(c10::MemoryFormat::ChannelsLast),
              w_scale[0],
              w_zp[0],
              std::nullopt)
        : at::native::fbgemm_utils::
              MakeEmptyAffineQuantizedChannelsLast3dTensor(
                  output_channels,
                  C_per_G,
                  kernel_d,
                  kernel_h,
                  kernel_w,
                  at::device(c10::kCPU).dtype(c10::kQInt8),
                  w_scale[0],
                  w_zp[0]);
  } else if (q_scheme == c10::kPerChannelAffine) {
    TORCH_CHECK(
        !transpose(),
        "Per Channel Quantization is currently disabled for transposed conv");
    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), at::device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), at::device(c10::kCPU).dtype(c10::kInt));
    unpacked_weights = kSpatialDim == 2
        ? at::_empty_per_channel_affine_quantized(
              {output_channels, C_per_G, kernel_h, kernel_w},
              scales.toType(c10::kDouble),
              zero_points.toType(c10::kLong),
              0, /* The output channel axis is 0 */
              at::device(c10::kCPU).dtype(c10::kQInt8),
              c10::MemoryFormat::ChannelsLast)
        : at::native::fbgemm_utils::
              MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
                  output_channels,
                  C_per_G,
                  kernel_d,
                  kernel_h,
                  kernel_w,
                  at::device(c10::kCPU).dtype(c10::kQInt8),
                  scales.toType(c10::kDouble),
                  zero_points.toType(c10::kLong));
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(q_scheme));
  }
  int8_t* unpacked_weights_p =
      reinterpret_cast<int8_t*>(unpacked_weights.data_ptr<c10::qint8>());
  packed_weights_p->unpack(unpacked_weights_p);
  if(transpose()){
    unpacked_weights =
        at::native::fbgemm_utils::TransposeConvTensorUnpackConversion<
            kSpatialDim>(unpacked_weights, groups);
  }
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(
      unpacked_weights, bias);
}

template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<
    2>::unpack();
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<
    3>::unpack();
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<
    kSpatialDim>::unpack() {
  TORCH_CHECK(
      kSpatialDim == 2,
      "QNNPACK only supports conv2d_unpack right "
      "now.");
  TORCH_CHECK(
        orig_weight.defined(),
        "Cannot unpack weights. "
        "Call at::globalContext()::setReleaseOriginalWeights(false) before packing or loading to enable unpacking.");
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(orig_weight, bias);
}

template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<
    2>::unpack();
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<
    3>::unpack();
#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<
    kSpatialDim>::unpack() {
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(
      orig_weight_.clone(), orig_bias_);
}

template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<
    2>::unpack();
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<
    3>::unpack();
#endif // #if AT_MKLDNN_ENABLED()

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `tuple`
- `vector`
- `ATen/ATen.h`
- `torch/library.h`
- `ATen/native/quantized/cpu/fbgemm_utils.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `ATen/native/quantized/cpu/OnednnUtils.h`
- `ATen/native/quantized/cpu/QuantUtils.h`
- `ATen/native/quantized/PackedParams.h`


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
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `qconv_unpack_impl.cpp_docs.md`
- **Keyword Index**: `qconv_unpack_impl.cpp_kw.md`
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

- **File Documentation**: `qconv_unpack_impl.cpp_docs.md_docs.md`
- **Keyword Index**: `qconv_unpack_impl.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
