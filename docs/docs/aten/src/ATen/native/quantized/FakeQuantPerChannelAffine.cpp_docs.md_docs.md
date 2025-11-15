# Documentation: `docs/aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp_docs.md`
- **Size**: 12,644 bytes (12.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp`
- **Size**: 9,894 bytes (9.66 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/FakeQuantAffine.h>

#include <c10/util/irange.h>

// FakeQuantize Op for PerChannelAffine quantization scheme.

namespace at::native {

// Use REGISTER_DISPATCH to run CPU and CUDA backend.
DEFINE_DISPATCH(fake_quant_per_channel_cachemask_stub);
DEFINE_DISPATCH(fake_quant_grad_learnable_channel_stub);

/* Per channel fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor.
  dY: Backward input tensor (_backward op only).
  scale: scale of per channel affine quantization
  zero_point: zero_point of per channel affine quantization
  axis: int specifying the axis to be quantized
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Fake quantized tensor (double dtype).

*/

Tensor fake_quantize_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
  auto res = at::fake_quantize_per_channel_affine_cachemask(
      self, scale, zero_point, axis, quant_min, quant_max);
  return std::get<0>(std::move(res));
}

std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float || scale.scalar_type() == at::kBFloat16,
              "Scale must be Float or BFloat16, found ", scale.scalar_type());
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Int || zero_point.scalar_type() == ScalarType::Float || zero_point.scalar_type() == ScalarType::Half,
              "Zero-point must be Int32, Float or Half, found ", zero_point.scalar_type());
  TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(
      scale.numel() == zero_point.numel(),
      "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(
      scale.numel() == self.size(axis),
      "dimensions of scale and zero-point are not consistent with input tensor")

  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");

  if(!at::isFloatingType(zero_point.scalar_type())){
      TORCH_CHECK(
          at::min(zero_point).item().toInt() >= quant_min &&
              at::max(zero_point).item().toInt() <= quant_max,
          "`zero_point` must be between `quant_min` and `quant_max`.");
  }
  TORCH_CHECK(
      axis >= 0 && axis <= self.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);

  c10::DimVector expected_shape(self.dim(), 1);
  expected_shape[axis] = self.size(axis);

  TensorIterator iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(Y)
    .add_input(self)
    .add_owned_input(native::_unsafe_view(scale, expected_shape))
    .add_owned_input(native::_unsafe_view(zero_point, expected_shape))
    .build();

  // TODO(future, optional): read once, write twice.  Not done at the moment
  //   for simplicity, as we do not expect this to be a bottleneck.
  TensorIterator iter_mask = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(mask)
    .add_input(self)
    .add_owned_input(native::_unsafe_view(scale, expected_shape))
    .add_owned_input(native::_unsafe_view(zero_point, expected_shape))
    .build();

  // TODO(future, optional): look into packing the mask further (BoolTensor uses
  //   1 byte per element, we only need 1 bit per element).
  fake_quant_per_channel_cachemask_stub(iter.device_type(), iter, iter_mask, quant_min, quant_max);
  return std::make_tuple(Y, mask);
}

/* Backward path to fake-quantize the 'inputs' tensor per channel, with mask.

Args:
  dY: output grad.
  mask: mask tensor from the forward pass.

Returns:
  dX (input grad).
*/
Tensor fake_quantize_per_channel_affine_cachemask_backward(
    const Tensor& dY,
    const Tensor& mask) {
  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool);
  TORCH_CHECK(mask.numel() == dY.numel(),
      "`mask` and `dY` are not the same size: ",
      "`mask` is size ", mask.numel(), " and `dY` is size ", dY.numel());
  if (dY.numel() <= 0) {
    return dY;
  }
  // Note: no additional kernels needed, since mask is pre-computed
  // and we can use the existing tensor multiplication kernels.
  return dY * mask;
}

static Tensor _get_rounded_zero_point(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // This assumes the per channel zero point vector is single-dimensioned.
  return zero_point.round().clamp_(quant_min, quant_max);
}

Tensor _fake_quantize_learnable_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  Tensor zero_point_rounded = _get_rounded_zero_point(zero_point, quant_min, quant_max).to(at::kInt);
  return native::fake_quantize_per_channel_affine(
    self, scale, zero_point_rounded, axis, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor> _fake_quantize_learnable_per_channel_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  /* The gradients for scale and zero point are calculated as below:
     Let Xfq be the fake quantized version of X.
     Let Xq be the quantized version of X (clamped at qmin and qmax).
     Let Delta and z be the scale and the zero point.
     :math:
      \frac{d\Delta }{dx} =
        \begin{cases}
          q_{\min} - z& \text{ if } X_q= q_{\min} \\
          q_{\max} - z& \text{ if } X_q= q_{\max} \\
          (X_{fq} - X) / \Delta & \text{ else }
        \end{cases}

      \frac{dz }{dx} =
        \begin{cases}
          -\Delta& \text{ if } X_q= q_{\min} \text{ or } X_q = q_{\max} \\
          0 & \text{ else }
        \end{cases}
  */
  bool is_bfloat16 = (X.scalar_type() == at::kBFloat16);
  at::Tensor X_ = is_bfloat16 ? X.to(ScalarType::Float) : X;
  at::Tensor dY_ = is_bfloat16 ? dY.to(ScalarType::Float) : dY;
  at::Tensor scale_ = is_bfloat16 ? scale.to(ScalarType::Float) : scale;
  at::Tensor zero_point_ = is_bfloat16 ? zero_point.to(ScalarType::Float) : zero_point;

  auto zero_point_rounded = _get_rounded_zero_point(zero_point_, quant_min, quant_max);

  TORCH_CHECK(dY_.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X_.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale_.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point_.scalar_type() == ScalarType::Float);

  TORCH_CHECK(X_.sizes() == dY_.sizes(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "Expecting `quant_min` <= 0 and `quant_max` >= 0");
  TORCH_CHECK(scale_.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point_.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(
      scale_.numel() == zero_point_.numel(),
      "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(
      scale_.numel() == X_.size(axis),
      "dimensions of scale and zero-point are not consistent with input tensor")

  TORCH_CHECK(
      at::min(zero_point_rounded).item().toLong() >= quant_min &&
          at::max(zero_point_rounded).item().toLong() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(
      axis >= 0 && axis < X_.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  if (X_.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  auto dX = at::empty_like(X_, X_.options(), MemoryFormat::Preserve);
  auto dScale_vec = at::empty_like(X_, X_.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X_, X_.options(), MemoryFormat::Preserve);
  auto numDimensions = X_.ndimension();

  // Create an axis mask for vectorizing and reshaping the scale and zero point tensors
  // into the same shapes as X along the channel axis.
  c10::DimVector axis_mask(numDimensions);
  for (const auto i : c10::irange(numDimensions)) {
    axis_mask[i] = (i == axis) ? X_.size(axis) : 1;
  }
  auto X_shape = X_.sizes();
  auto scale_vectorized = scale_.reshape(at::IntArrayRef(axis_mask.data(), numDimensions)).expand(X_shape);
  auto zero_point_vectorized = zero_point_rounded.reshape(at::IntArrayRef(axis_mask.data(), numDimensions)).expand(X_shape);

  auto iter = TensorIteratorConfig()
    .add_output(dX)
    .add_output(dScale_vec)
    .add_output(dZeroPoint_vec)
    .add_input(X_)
    .add_input(dY_)
    .add_input(scale_vectorized)
    .add_input(zero_point_vectorized)
    .build();

  fake_quant_grad_learnable_channel_stub(
    X_.device().type(), iter, quant_min, quant_max, grad_factor);

  auto numElements = X_.ndimension() - 1;

  // Create a collection of axes that include all but the channel axis for
  // reduction when summing over the dScale and dZeroPoint tensors.
  c10::DimVector axis_for_reduction(numElements);
  for (const auto i : c10::irange(axis)) {
    axis_for_reduction[i] = i;
  }
  for (const auto i : c10::irange(axis, numElements)) {
    axis_for_reduction[i] = i + 1;
  }

  auto dScale = dScale_vec.sum(at::IntArrayRef(axis_for_reduction.data(), numElements));
  auto dZeroPoint = dZeroPoint_vec.sum(at::IntArrayRef(axis_for_reduction.data(), numElements));

  return std::make_tuple(dX, dScale, dZeroPoint);
}
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/Dispatch.h`
- `ATen/NativeFunctions.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/cpu/Loops.h`
- `ATen/native/quantized/FakeQuantAffine.h`
- `c10/util/irange.h`


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

- **File Documentation**: `FakeQuantPerChannelAffine.cpp_docs.md`
- **Keyword Index**: `FakeQuantPerChannelAffine.cpp_kw.md`
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

- **File Documentation**: `FakeQuantPerChannelAffine.cpp_docs.md_docs.md`
- **Keyword Index**: `FakeQuantPerChannelAffine.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
