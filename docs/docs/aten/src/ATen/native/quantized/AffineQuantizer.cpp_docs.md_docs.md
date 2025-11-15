# Documentation: `docs/aten/src/ATen/native/quantized/AffineQuantizer.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/AffineQuantizer.cpp_docs.md`
- **Size**: 12,177 bytes (11.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/AffineQuantizer.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/AffineQuantizer.cpp`
- **Size**: 9,649 bytes (9.42 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/quantized/AffineQuantizer.h>


namespace at::native {

DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_sub_byte_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_sub_byte_stub);

namespace {

void checkRoundingMode(const std::string& fn_name) {
  // Disabling this warning message for now as it is printed incorrectly. Need
  // to fix

  /*  TORCH_WARN_ONCE(
        std::fegetround() != FE_TONEAREST,
        fn_name,
        " current rounding mode is not set to round-to-nearest-ties-to-even
     (FE_TONEAREST). This will cause accuracy issues in quantized models.");
  */
  return;
}

void checkFloatTensor(const std::string& fn_name, const Tensor& t) {
  TORCH_CHECK(
      t.scalar_type() == kFloat, fn_name, " expects a Float Tensor, got ",
      t.scalar_type());
}

void checkSameDevice(
    const std::string& fn_name,
    const Tensor& t1,
    const Tensor& t2) {
  TORCH_CHECK(
      t1.device() == t2.device(),
      fn_name,
      " expects a quantized and float tensors to be on the same device.");
}

template <typename T>
void checkQuantizedTensor(const std::string& fn_name, const Tensor& t) {
  TORCH_CHECK(t.is_quantized(), fn_name, " expects a quantized Tensor.");
  TORCH_CHECK(
      t.scalar_type() == caffe2::TypeMeta::Make<T>(),
      fn_name,
      " expects a ",
      caffe2::TypeMeta::Make<T>(),
      " Tensor, got ",
      t.scalar_type());
}

template <typename T>
void checkZeroPoint(const std::string& fn_name, int64_t zero_point) {
  TORCH_CHECK(
      zero_point <= std::numeric_limits<T>::max(),
      fn_name,
      " zero_point ",
      zero_point,
      " is above upper bound.");
  TORCH_CHECK(
      zero_point >= std::numeric_limits<T>::min(),
      fn_name,
      " zero_point ",
      zero_point,
      " is below lower bound.");
}

template <typename T>
void checkZeroPoints(const std::string& fn_name, const Tensor& zero_points) {
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  for (const auto i : c10::irange(zero_points.numel())) {
    checkZeroPoint<T>(fn_name, zero_points_data[i]);
  }
}

void checkSameSize(
    const std::string& fn_name,
    const Tensor& qt,
    const Tensor& rt) {
  TORCH_CHECK(
      qt.sizes().equals(rt.sizes()),
      fn_name,
      " only works with Tensors with the same shape");
}

void checkPerChannelParamsSize(
    const Tensor& rtensor,
    int64_t axis,
    const Tensor& scales,
    const Tensor& zero_points
) {
  int64_t channel = rtensor.size(axis);
  TORCH_CHECK(
      channel == int64_t(scales.numel()),
      "length of scales must equal to channel, expected ", channel, " got, ", scales.numel());
  TORCH_CHECK(
      channel == int64_t(zero_points.numel()),
      "length of zero_points must equal to channel expected ", channel, " got, ", zero_points.numel());
}

} // anonymous namespace

Tensor& quantize_tensor_per_tensor_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  static constexpr auto fn_name = "quantize_tensor_per_tensor_affine";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  // Temporary solution to pack the tensor if dtype is torch.quint4x2
  // Can move this into the fbgemm::Quantize op.
  if (qtensor.scalar_type() == at::ScalarType::QUInt4x2 || qtensor.scalar_type() == at::ScalarType::QUInt2x4) {
    quantize_tensor_per_tensor_affine_sub_byte_stub(
        rtensor.device().type(), rtensor, qtensor, scale, zero_point);
  } else {
    quantize_tensor_per_tensor_affine_stub(
        rtensor.device().type(), rtensor, qtensor, scale, zero_point);
  }
  return qtensor;
}

Tensor& quantize_tensor_per_channel_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    Tensor zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "quantize_tensor_per_channel_affine";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    if (qtensor.device().type() != c10::DeviceType::CUDA &&
        qtensor.device().type() != c10::DeviceType::XPU &&
        qtensor.device().type() != c10::DeviceType::PrivateUse1) {
      checkZeroPoints<underlying_t>(fn_name, zero_points);
    }  // for cuda and privateuse1, this check will occur in the actual device function
  });

  TORCH_CHECK(
      0 <= axis && axis < rtensor.dim(),
      "Channel axis out of range in per channel affine quantization. Got: ",
      axis,
      "Expected: [0, ",
      rtensor.dim(),
      ")");
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  quantize_tensor_per_channel_affine_stub(
      rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
  return qtensor;
}

Tensor& quantize_tensor_per_channel_float_qparams(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name =
      "quantize_tensor_per_channel_float_qparams";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
  });

  TORCH_CHECK(
      0 <= axis && axis < rtensor.dim(),
      "Channel axis out of range in per channel float qparams quantization. Got: ",
      axis,
      "Expected: [0, ",
      rtensor.dim(),
      ")");
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  quantize_tensor_per_channel_float_qparams_stub(
      rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
  return qtensor;
}

Tensor& dequantize_tensor_per_tensor_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  static constexpr auto fn_name = "dequantize_tensor_per_tensor_affine";
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  if (qtensor.scalar_type() == at::ScalarType::QUInt4x2 || qtensor.scalar_type() == at::ScalarType::QUInt2x4) {
    dequantize_tensor_per_tensor_affine_sub_byte_stub(
        qtensor.device().type(), qtensor, rtensor, scale, zero_point);
  } else {
    dequantize_tensor_per_tensor_affine_stub(
        qtensor.device().type(), qtensor, rtensor, scale, zero_point);
  }
  return rtensor;
}

Tensor& dequantize_tensor_per_channel_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    Tensor zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine";

  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    if(qtensor.device().type() != c10::DeviceType::CUDA &&
       qtensor.device().type() != c10::DeviceType::XPU &&
       qtensor.device().type() != c10::DeviceType::PrivateUse1){
      checkZeroPoints<underlying_t>(fn_name, zero_points);
    }  // for cuda and privateuse1, this check will occur in the actual device function
  });

  TORCH_CHECK(
      0 <= axis && axis < qtensor.dim(),
      "Channel axis out of range in per channel affine dequantization. Got:",
      axis,
      " Expected: [0, ",
      qtensor.dim(),
      ")");
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  dequantize_tensor_per_channel_affine_stub(
      qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
  return rtensor;
}

Tensor& dequantize_tensor_per_channel_float_qparams(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine";

  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
  });

  TORCH_CHECK(
      0 <= axis && axis < qtensor.dim(),
      "Channel axis out of range in per channel float qparams dequantization. Got:",
      axis,
      " Expected: [0, ",
      qtensor.dim(),
      ")");
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  dequantize_tensor_per_channel_float_qparams_stub(
      qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
  return rtensor;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/quantized/AffineQuantizer.h`


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

Files in the same folder (`aten/src/ATen/native/quantized`):

- [`QTensor.cpp_docs.md`](./QTensor.cpp_docs.md)
- [`FakeQuantAffine.h_docs.md`](./FakeQuantAffine.h_docs.md)
- [`qconv_unpack.cpp_docs.md`](./qconv_unpack.cpp_docs.md)
- [`AffineQuantizerBase.h_docs.md`](./AffineQuantizerBase.h_docs.md)
- [`AffineQuantizerBase.cpp_docs.md`](./AffineQuantizerBase.cpp_docs.md)
- [`PackedParams.h_docs.md`](./PackedParams.h_docs.md)
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)
- [`AffineQuantizer.h_docs.md`](./AffineQuantizer.h_docs.md)
- [`qlinear_unpack.cpp_docs.md`](./qlinear_unpack.cpp_docs.md)
- [`TensorFactories.cpp_docs.md`](./TensorFactories.cpp_docs.md)


## Cross-References

- **File Documentation**: `AffineQuantizer.cpp_docs.md`
- **Keyword Index**: `AffineQuantizer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized`):

- [`qconv_unpack.cpp_kw.md_docs.md`](./qconv_unpack.cpp_kw.md_docs.md)
- [`TensorCompare.cpp_docs.md_docs.md`](./TensorCompare.cpp_docs.md_docs.md)
- [`QTensor.cpp_docs.md_docs.md`](./QTensor.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`PackedParams.h_docs.md_docs.md`](./PackedParams.h_docs.md_docs.md)
- [`AffineQuantizerBase.cpp_docs.md_docs.md`](./AffineQuantizerBase.cpp_docs.md_docs.md)
- [`FakeQuantAffine.h_kw.md_docs.md`](./FakeQuantAffine.h_kw.md_docs.md)
- [`FakeQuantPerChannelAffine.cpp_kw.md_docs.md`](./FakeQuantPerChannelAffine.cpp_kw.md_docs.md)
- [`TensorCompare.cpp_kw.md_docs.md`](./TensorCompare.cpp_kw.md_docs.md)
- [`Copy.h_kw.md_docs.md`](./Copy.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `AffineQuantizer.cpp_docs.md_docs.md`
- **Keyword Index**: `AffineQuantizer.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
