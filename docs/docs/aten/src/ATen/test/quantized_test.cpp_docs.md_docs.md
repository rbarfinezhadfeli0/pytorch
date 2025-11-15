# Documentation: `docs/aten/src/ATen/test/quantized_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/quantized_test.cpp_docs.md`
- **Size**: 14,516 bytes (14.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/quantized_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/quantized_test.cpp`
- **Size**: 11,768 bytes (11.49 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/test/test_assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
// For quantize_val
#include <ATen/native/quantized/AffineQuantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <ATen/quantized/Quantizer.h>

using namespace at;
#ifndef ATEN_CPU_STATIC_DISPATCH

TEST(TestQTensor, QuantDequantAPIs) {
  auto num_elements = 10;
  Tensor r = at::ones({num_elements});
  const double scale = 1.0;
  const int64_t zero_point = 2;
  const Tensor qr = at::quantize_per_tensor(r, scale, zero_point, kQUInt8);
  ASSERT_EQ(qr.q_scale(), scale);
  ASSERT_EQ(qr.q_zero_point(), zero_point);
  ASSERT_TRUE(qr.is_quantized());
  ASSERT_FALSE(r.is_quantized());

  // int_repr
  Tensor int_repr = qr.int_repr();
  auto* int_repr_data = int_repr.data_ptr<uint8_t>();
  for (const auto i : c10::irange(num_elements)) {
    ASSERT_EQ(int_repr_data[i], 3);
  }

  // Check for correct quantization
  auto r_data = r.data_ptr<float>();
  auto qr_data = qr.data_ptr<quint8>();
  for (const auto i : c10::irange(num_elements)) {
    ASSERT_EQ(
        native::quantize_val<quint8>(scale, zero_point, r_data[i]).val_,
        qr_data[i].val_);
  }

  // Check for correct dequantization
  Tensor rqr = qr.dequantize();
  auto rqr_data = rqr.data_ptr<float>();
  for (const auto i : c10::irange(num_elements)) {
    ASSERT_EQ(r_data[i], rqr_data[i]);
  }
  for (const auto i : c10::irange(num_elements)) {
    ASSERT_EQ(
        r_data[i],
        native::dequantize_val(qr.q_scale(), qr.q_zero_point(), qr_data[i]));
  }

  // Check for correct requantization
  double new_scale = 2.0;
  int64_t new_zero_point = 1;
  Tensor reqr = at::quantize_per_tensor(r, new_scale, new_zero_point, kQInt8);
  auto reqr_data = reqr.data_ptr<qint8>();
  for (const auto i : c10::irange(num_elements)) {
    reqr_data[i].val_ =
        native::requantize_val<quint8, qint8>(
            scale, zero_point, new_scale, new_zero_point, qr_data[i])
            .val_;
    const qint8 expected =
        native::quantize_val<qint8>(new_scale, new_zero_point, rqr_data[i]);
    ASSERT_EQ(expected.val_, reqr_data[i].val_);
  }
}

TEST(TestQTensor, RoundingMode) {
  // We assume that quantization is defined as:
  //   qx = clamp(zero_point + round(x / scale))
  // If the zero_point is added before rounding, the result will be wrong.
  int32_t zero_point = 5;
  std::vector<float> x_values{
      -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
  std::vector<uint8_t> qx_expect{
      0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11}; // scale = 1.0

  Tensor x = from_blob(x_values.data(), x_values.size());
  Tensor qx = at::quantize_per_tensor(x, /*scale=*/1.0, zero_point, kQUInt8);

  auto qx_data = qx.data_ptr<quint8>();
  for (const auto idx : c10::irange(x_values.size())) {
    ASSERT_EQ(qx_expect[idx], qx_data[idx].val_)
        << "Tie breaking during rounding element " << idx << " failed!";
  }
}

TEST(TestQTensor, Item) {
  Tensor r = at::ones({1});
  const float scale = 1;
  const int32_t zero_point = 2;
  Tensor qr = at::quantize_per_tensor(r, scale, zero_point, kQUInt8);
  ASSERT_EQ(r.item().to<float>(), qr.item().to<float>());
}

TEST(TestQTensor, EmptyQuantized) {
  float scale = 0.5;
  int zero_point = 10;
  int val = 100;
  int numel = 10;
  Tensor q = at::_empty_affine_quantized(
      {numel}, at::device(at::kCPU).dtype(kQUInt8), scale, zero_point);
  // Assigning to QTensor
  auto* q_data = q.data_ptr<quint8>();
  for (const auto i : c10::irange(numel)) {
    q_data[i].val_ = val;
  }

  // dequantize
  auto r = q.dequantize();
  auto* r_data = r.data_ptr<float>();
  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ(r_data[i], (val - zero_point) * scale);
  }
}

TEST(TestQTensor, EmptyPerchannelQuantized) {
  int numel = 10;
  auto scales = rand({numel}).toType(kDouble);
  auto zero_points = randint(10, {10}).toType(kLong);
  int val = 100;
  int ch_axis = 0;
  Tensor q = at::_empty_per_channel_affine_quantized(
      {numel},
      scales,
      zero_points,
      ch_axis,
      at::device(at::kCPU).dtype(kQUInt8));
  // Assigning to QTensor
  auto* q_data = q.data_ptr<quint8>();
  for (const auto i : c10::irange(numel)) {
    q_data[i].val_ = val;
  }

  // dequantize
  auto r = q.dequantize();
  auto* r_data = r.data_ptr<float>();
  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ(
        r_data[i],
        (val - zero_points[i].item().to<int>()) * scales[i].item().to<float>());
  }
}

TEST(TestQTensor, QuantizePerChannel4d) {
  int C = 64, H = 10, W = 10;
  auto scales = rand({C}).toType(kDouble);
  auto zero_points = randint(10, {C}).toType(kLong);
  int ch_axis = 1;
  // create 4d tensor where each H x W image is a range(0, H*W)
  Tensor tensor = at::empty({1, C, H, W}, at::device(at::kCPU).dtype(kFloat));
  auto* tensor_data = tensor.mutable_data_ptr<float>();
  for (int c = 0, i = 0; c < C; ++c) {
    for (int e = 0; e < H * W; ++e, ++i) {
      tensor_data[i] = e;
    }
  }
  // quantize and check values
  Tensor q = at::native::quantize_per_channel(
      tensor, scales, zero_points, ch_axis, kQUInt8);
  auto* q_data = (uint8_t*)q.data_ptr<quint8>();
  for (int c = 0, i = 0; c < C; ++c) {
    float inv_scale = 1.0f / static_cast<float>(scales[c].item<double>());
    int64_t zero_point = zero_points[c].item<int64_t>();
    for (int e = 0; e < H * W; ++e, ++i) {
      // downsize qval to 255 if val is greater than max uint8_t value
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers,bugprone-narrowing-conversions)
      int qval = std::min<int>(zero_point + std::nearbyint(e * inv_scale), 255);
      ASSERT_EQ((int)q_data[i], qval);
    }
  }
}

TEST(TestQTensor, QuantizePerChannel4dChannelsLast) {
  int C = 64, H = 10, W = 10;
  auto scales = rand({C}).toType(kDouble);
  auto zero_points = randint(10, {C}).toType(kLong);
  int ch_axis = 1;
  // create 4d tensor where each H x W image is a range(0, H*W)
  Tensor tensor = at::empty(
      {1, C, H, W},
      at::device(at::kCPU).dtype(kFloat).memory_format(
          at::MemoryFormat::ChannelsLast));
  auto* tensor_data = tensor.data_ptr<float>();
  for (int e = 0, i = 0; e < H * W; ++e) {
    for (int c = 0; c < C; ++c, ++i) {
      tensor_data[i] = e;
    }
  }

  // quantize and check values
  Tensor q = at::native::quantize_per_channel(
      tensor, scales, zero_points, ch_axis, kQUInt8);
  auto* q_data = (uint8_t*)q.data_ptr<quint8>();
  for (int e = 0, i = 0; e < H * W; ++e) {
    for (int c = 0; c < C; ++c, ++i) {
      float inv_scale = 1.0f / static_cast<float>(scales[c].item<double>());
      int64_t zero_point = zero_points[c].item<int64_t>();
      // downsize qval to 255 if val is greater than max uint8_t value
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers,bugprone-narrowing-conversions)
      int qval = std::min<int>(zero_point + std::nearbyint(e * inv_scale), 255);
      ASSERT_EQ((int)q_data[i], qval);
    }
  }
}

TEST(TestQTensor, FromBlobQuantizedPerTensor) {
  const float scale = 0.1;
  const int64_t zero_point = 10;
  std::vector<int64_t> shape = {5, 10};
  auto numel = c10::multiply_integers(shape);

  TensorOptions options(at::kQUInt8);

  auto custom_vec = std::make_unique<std::vector<uint8_t>>();
  custom_vec->resize(numel);

  uint8_t* custom_data = custom_vec->data();
  for (const auto i : c10::irange(numel)) {
    custom_data[i] = i;
  }
  bool customDataDeleted{false};
  auto deleteWhenDone = custom_vec.release();
  auto deleter = [deleteWhenDone, custom_data, &customDataDeleted](void* inp) {
    ASSERT_EQ((void*)inp, (void*)custom_data);
    delete deleteWhenDone;
    customDataDeleted = true;
  };
  {
  Tensor qtensor = at::from_blob_quantized_per_tensor_affine(custom_data, shape, deleter, scale, zero_point, options);

  uint8_t* q_data = (uint8_t*)qtensor.data_ptr<quint8>();
  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ((int)custom_data[i], (int)q_data[i]);
  }
  for (int h = 0, i = 0; h < shape[0]; ++h) {
    for (int w = 0; w < shape[1]; ++w, ++i) {
      ASSERT_EQ(
          qtensor[h][w].item<float>(),
          (custom_data[i] - zero_point) * scale);
    }
  }
  ASSERT_EQ((float)qtensor.q_scale(), (float)scale);
  ASSERT_EQ(qtensor.q_zero_point(), zero_point);
  }
  TORCH_CHECK(customDataDeleted);
}

TEST(TestQTensor, FromBlobQuantizedPerChannel) {
  int C = 64, H = 10, W = 5;
  std::vector<int64_t> shape = {1, C, H, W};
  auto scales = rand({C}).toType(kDouble);
  auto zero_points = randint(10, {C}).toType(kLong);
  auto numel = c10::multiply_integers(shape);
  int ch_axis = 1;
  TensorOptions options(at::kQUInt8);

  auto custom_vec = std::make_unique<std::vector<uint8_t>>();
  custom_vec->resize(numel);

  uint8_t* custom_data = custom_vec->data();
  for (const auto i : c10::irange(numel)) {
    custom_data[i] = i;
  }
  bool customDataDeleted{false};
  auto deleteWhenDone = custom_vec.release();
  auto deleter = [deleteWhenDone, custom_data, &customDataDeleted](void* inp) {
    ASSERT_EQ((void*)inp, (void*)custom_data);
    delete deleteWhenDone;
    customDataDeleted = true;
  };
  {
  Tensor qtensor = at::from_blob_quantized_per_channel_affine(custom_data, shape, deleter, scales, zero_points, ch_axis, options);
  uint8_t* q_data = (uint8_t*)qtensor.data_ptr<quint8>();
  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ((int)custom_data[i], (int)q_data[i]);
  }
  ASSERT_TRUE(at::allclose(qtensor.q_per_channel_scales(), scales));
  ASSERT_TRUE(at::allclose(qtensor.q_per_channel_zero_points(), zero_points));
  ASSERT_TRUE(qtensor.is_quantized());
  }
  TORCH_CHECK(customDataDeleted);
}

#if defined(__ARM_NEON__) || defined(__aarch64__)
TEST(TestQTensor, TestArmVectorizedQuantizeDequantize) {
  const float scale = 7;
  const int numel = 132;

  std::vector<float> x_values;
  for (const auto i : c10::irange(numel)) {
    x_values.push_back(9 * i);
  }

  const Tensor x = from_blob(x_values.data(), x_values.size());

  auto test_for_datatype = [&](
      const ScalarType scalar_type,
      const auto get_data_ptr,
      const auto quantize_val_with_datatype,
      const int zero_point_min,
      const int zero_point_max) {
    for (int zero_point : {zero_point_min, 10, zero_point_max}) {
      const Tensor q = at::quantize_per_tensor(x, scale, zero_point, scalar_type);
      auto* q_data = get_data_ptr(q);
      for (const auto i : c10::irange(numel)) {
        ASSERT_EQ(
          q_data[i].val_,
          quantize_val_with_datatype(scale, zero_point, x_values[i]).val_);
      }
      const Tensor r = q.dequantize();
      const float* r_data = r.const_data_ptr<float>();
      for (const auto i : c10::irange(numel)) {
        ASSERT_FLOAT_EQ(
          r_data[i],
          native::dequantize_val(scale, zero_point, q_data[i]));
      }
    }
  };

  // Unsigned Int 8
  test_for_datatype(
    kQUInt8,
    [](Tensor q) { return q.data_ptr<quint8>(); },
    native::quantize_val<quint8>,
    std::numeric_limits<uint8_t>::min(),
    std::numeric_limits<uint8_t>::max());

  // Signed Int 8
  test_for_datatype(
    kQInt8,
    [](Tensor q) { return q.data_ptr<qint8>(); },
    native::quantize_val<qint8>,
    std::numeric_limits<int8_t>::min(),
    std::numeric_limits<int8_t>::max());

  // Signed Int 32 (not optimized with vectorization)
  test_for_datatype(
    kQInt32,
    [](Tensor q) { return q.data_ptr<qint32>(); },
    native::quantize_val<qint32>,
    std::numeric_limits<int32_t>::min(),
    std::numeric_limits<int32_t>::max());
}
#endif // (__ARM_NEON__) || defined(__aarch64__)

#endif // ATEN_CPU_STATIC_DISPATCH

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`
- `ATen/test/test_assert.h`
- `cmath`
- `iostream`
- `limits`
- `memory`
- `sstream`
- `type_traits`
- `ATen/native/quantized/AffineQuantizer.h`
- `c10/core/ScalarType.h`
- `c10/util/irange.h`
- `ATen/quantized/Quantizer.h`


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

This is a test file. Run it with:

```bash
python aten/src/ATen/test/quantized_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `quantized_test.cpp_docs.md`
- **Keyword Index**: `quantized_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

This is a test file. Run it with:

```bash
python docs/aten/src/ATen/test/quantized_test.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `quantized_test.cpp_docs.md_docs.md`
- **Keyword Index**: `quantized_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
