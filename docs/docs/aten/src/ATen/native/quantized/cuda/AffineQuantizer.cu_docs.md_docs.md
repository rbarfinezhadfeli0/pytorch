# Documentation: `docs/aten/src/ATen/native/quantized/cuda/AffineQuantizer.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cuda/AffineQuantizer.cu_docs.md`
- **Size**: 12,061 bytes (11.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cuda/AffineQuantizer.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cuda/AffineQuantizer.cu`
- **Size**: 9,417 bytes (9.20 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <cmath>
#include <ATen/native/cuda/Loops.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/any.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/lt.h>
#endif

namespace at::native {
namespace {

template <typename T>
void check_zero_points_cuda(
    const std::string& fn_name,
    const Tensor& zero_points) {
  constexpr int64_t qmin = std::numeric_limits<T>::min();
  constexpr int64_t qmax = std::numeric_limits<T>::max();
  auto zp_within_upper = at::any(at::gt(zero_points, qmax)).item().equal(false);
  auto zp_within_lower = at::any(at::lt(zero_points, qmin)).item().equal(false);
  TORCH_CHECK(
    zp_within_lower,
    fn_name,
    "zero_point is below lower bound.");
  TORCH_CHECK(
    zp_within_upper,
    fn_name,
    "zero_point is above upper bound.");
}

void quantize_tensor_per_tensor_affine_cuda(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cuda", [&]() {
        constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
        constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();

        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(qtensor)
                        .add_input(rtensor)
                        .add_input(qtensor)
                        .build();
        gpu_kernel(
            iter,
            [=] GPU_LAMBDA(float raw_val, scalar_t quantized_val) -> scalar_t {
              int64_t qvalue =
                  static_cast<int64_t>(std::nearbyint(raw_val / scale) + zero_point);
              qvalue = std::max<int64_t>(qvalue, qmin);
              qvalue = std::min<int64_t>(qvalue, qmax);
              quantized_val.val_ = qvalue;
              return quantized_val;
            });
      });
}

void dequantize_tensor_per_tensor_affine_cuda(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cuda", [&]() {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(rtensor)
                        .add_input(qtensor)
                        .build();
        gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t value) -> float {
          return (static_cast<float>(value.val_) - zero_point) * scale;
        });
      });
}

void quantize_tensor_per_channel_affine_cuda(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "quantize_tensor_per_channel_affine_cuda";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(qtensor)
                  .add_input(rtensor)
                  .add_input(qtensor)
                  .add_input(shaped_scales)
                  .add_input(shaped_zero_points)
                  .build();

  AT_DISPATCH_QINT_TYPES(
    qtensor.scalar_type(), fn_name, [&]() {
      check_zero_points_cuda<underlying_t>(fn_name, zero_points);

      constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
      constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();
      // trying to match _quantize_per_channel_ref_nd in test_quantized_tensor.py
      gpu_kernel(
          iter,
          [=] GPU_LAMBDA(float raw_val, scalar_t quantized_val, double scale, int64_t zero_point) -> scalar_t {

            int64_t qvalue =
                static_cast<int64_t>(std::nearbyint(raw_val/scale) + zero_point);
            qvalue = std::max<int64_t>(qvalue, qmin);
            qvalue = std::min<int64_t>(qvalue, qmax);
            quantized_val.val_ = qvalue;
            return quantized_val;
          });
    });
}

void dequantize_tensor_per_channel_affine_cuda(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine_cuda";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(),
      fn_name,
      [&]() {
        check_zero_points_cuda<underlying_t>(fn_name, zero_points);

        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(rtensor)
                        .add_input(qtensor)
                        .add_input(shaped_scales)
                        .add_input(shaped_zero_points)
                        .build();

        gpu_kernel(
            iter,
            [=] GPU_LAMBDA(
                scalar_t value, double scale, int64_t zero_point) -> float {
              return static_cast<float>(value.val_ - zero_point) * scale;
            });
      });
}

void quantize_tensor_per_channel_float_qparams_cuda(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "quantize_tensor_per_channel_float_qparams_cuda";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(qtensor)
                  .add_input(rtensor)
                  .add_input(qtensor)
                  .add_input(shaped_scales)
                  .add_input(shaped_zero_points)
                  .build();

  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(),
      fn_name,
      [&]() {
        check_zero_points_cuda<underlying_t>(fn_name, zero_points);

        constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
        constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();
        // trying to match _quantize_per_channel_ref_nd in
        gpu_kernel(
            iter,
            [=] GPU_LAMBDA(
                float raw_val,
                scalar_t quantized_val,
                float scale,
                float zero_point) -> scalar_t {
              float inv_scale = 1.0f / scale;
              int64_t qvalue = lrintf(raw_val * inv_scale + zero_point);
              qvalue = std::max<int64_t>(qvalue, qmin);
              qvalue = std::min<int64_t>(qvalue, qmax);
              quantized_val.val_ = qvalue;
              return quantized_val;
            });
      });
}

void dequantize_tensor_per_channel_float_qparams_cuda(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_float_qparams_cuda";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(),
      fn_name,
      [&]() {
        check_zero_points_cuda<underlying_t>(fn_name, zero_points);

        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(rtensor)
                        .add_input(qtensor)
                        .add_input(shaped_scales)
                        .add_input(shaped_zero_points)
                        .build();

        gpu_kernel(
            iter,
            [=] GPU_LAMBDA(
                scalar_t value, float scale, float zero_point) -> float {
              return (static_cast<float>(value.val_) - zero_point) * scale;
            });
      });
}

} // anonymous namespace

REGISTER_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &quantize_tensor_per_tensor_affine_cuda);
REGISTER_DISPATCH(
    dequantize_tensor_per_tensor_affine_stub,
    &dequantize_tensor_per_tensor_affine_cuda);
REGISTER_DISPATCH(
    quantize_tensor_per_channel_affine_stub,
    &quantize_tensor_per_channel_affine_cuda);
REGISTER_DISPATCH(
    dequantize_tensor_per_channel_affine_stub,
    &dequantize_tensor_per_channel_affine_cuda);
REGISTER_DISPATCH(
    quantize_tensor_per_channel_float_qparams_stub,
    &quantize_tensor_per_channel_float_qparams_cuda);
REGISTER_DISPATCH(
    dequantize_tensor_per_channel_float_qparams_stub,
    &dequantize_tensor_per_channel_float_qparams_cuda);
} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/quantized/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `REGISTER_DISPATCH`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/TensorIterator.h`
- `ATen/native/quantized/AffineQuantizer.h`
- `cmath`
- `ATen/native/cuda/Loops.cuh`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_unsafe_view_native.h`
- `ATen/ops/any.h`
- `ATen/ops/gt.h`
- `ATen/ops/lt.h`


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

Files in the same folder (`aten/src/ATen/native/quantized/cuda`):

- [`FakeQuantizeCore.cu_docs.md`](./FakeQuantizeCore.cu_docs.md)
- [`IntReprQuant.cu_docs.md`](./IntReprQuant.cu_docs.md)
- [`Activation.cu_docs.md`](./Activation.cu_docs.md)
- [`MakePerTensorQuantizedTensor.cu_docs.md`](./MakePerTensorQuantizedTensor.cu_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`FusedObsFakeQuant.cu_docs.md`](./FusedObsFakeQuant.cu_docs.md)
- [`EmbeddingBag.cu_docs.md`](./EmbeddingBag.cu_docs.md)


## Cross-References

- **File Documentation**: `AffineQuantizer.cu_docs.md`
- **Keyword Index**: `AffineQuantizer.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cuda`):

- [`FusedObsFakeQuant.cu_docs.md_docs.md`](./FusedObsFakeQuant.cu_docs.md_docs.md)
- [`FakeQuantizeCore.cu_kw.md_docs.md`](./FakeQuantizeCore.cu_kw.md_docs.md)
- [`FusedObsFakeQuant.cu_kw.md_docs.md`](./FusedObsFakeQuant.cu_kw.md_docs.md)
- [`Activation.cu_docs.md_docs.md`](./Activation.cu_docs.md_docs.md)
- [`AffineQuantizer.cu_kw.md_docs.md`](./AffineQuantizer.cu_kw.md_docs.md)
- [`EmbeddingBag.cu_kw.md_docs.md`](./EmbeddingBag.cu_kw.md_docs.md)
- [`EmbeddingBag.cu_docs.md_docs.md`](./EmbeddingBag.cu_docs.md_docs.md)
- [`IntReprQuant.cu_kw.md_docs.md`](./IntReprQuant.cu_kw.md_docs.md)
- [`IntReprQuant.cu_docs.md_docs.md`](./IntReprQuant.cu_docs.md_docs.md)


## Cross-References

- **File Documentation**: `AffineQuantizer.cu_docs.md_docs.md`
- **Keyword Index**: `AffineQuantizer.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
