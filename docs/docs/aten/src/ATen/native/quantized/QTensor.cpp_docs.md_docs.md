# Documentation: `docs/aten/src/ATen/native/quantized/QTensor.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/QTensor.cpp_docs.md`
- **Size**: 15,453 bytes (15.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/QTensor.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/QTensor.cpp`
- **Size**: 12,830 bytes (12.53 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

#include <c10/util/irange.h>

#include <cmath>
#include <utility>


namespace at::native {

Tensor quantize_per_tensor_dynamic(
    const Tensor& self,
    ScalarType dtype,
    bool reduce_range) {
  TORCH_CHECK( (dtype == ScalarType::QInt8 || dtype == ScalarType::QUInt8 || dtype == ScalarType::Half), "dtype ", dtype, "not supported");
  auto input_contig = self.contiguous();
  if (dtype == ScalarType::Half) {
    return input_contig.to(ScalarType::Half);
  }
  float x_min = input_contig.min().item<float>();
  float x_max = input_contig.max().item<float>();

  if (reduce_range && at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    reduce_range = false;
  }

  int qmin = 0;
  int qmax = 0;

  if (dtype == ScalarType::QInt8) {
    qmin = -128;
    qmax = 127;
  } else {
    // for now, this branch executes for dtype == ScalarType::QUInt8
    // additional cases will be added when quantization support for other dtypes becomes available
    qmin = 0;
    qmax = 255;
  }

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/qmin,
      /*qmax=*/qmax,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  return at::native::quantize_per_tensor(self, q_params.scale, q_params.zero_point, dtype);
}

Tensor quantize_per_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

Tensor quantize_per_tensor_tensor_qparams(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    ScalarType dtype) {
  auto quantizer = make_per_tensor_affine_quantizer(scale.item().toDouble(), zero_point.item().toLong(), dtype);
  return quantizer->quantize(self);
}

std::vector<Tensor> quantize_per_tensor_list_cpu(
    TensorList tensors,
    const Tensor& scales,
    const Tensor& zero_points,
    ScalarType dtype) {
  std::vector<Tensor> quantized_tensors;
  for (const auto i : c10::irange(tensors.size())) {
    quantized_tensors.push_back(at::quantize_per_tensor(
        tensors[i],
        scales[static_cast<int64_t>(i)].item<double>(),
        zero_points[static_cast<int64_t>(i)].item<int64_t>(),
        dtype));
  }
  return quantized_tensors;
}

Tensor quantize_per_channel(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  auto quantizer = make_per_channel_affine_quantizer(scales, zero_points, axis, dtype);
  return quantizer->quantize(self);
}

Tensor dequantize_cpu_or_cuda(const Tensor& self) {
  return self.to(at::kFloat);
}

Tensor dequantize_quantized(const Tensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

std::vector<Tensor> dequantize_tensors_quantized_cpu(TensorList tensors) {
  std::vector<Tensor> dequantized_tensors;
  for (const auto & tensor : tensors) {
    dequantized_tensors.push_back(tensor.dequantize());
  }
  return dequantized_tensors;
}

double q_scale_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
  return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->scale();
}

int64_t q_zero_point_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
  return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->zero_point();
}

Tensor q_per_channel_scales(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->scales();
}

Tensor q_per_channel_zero_points(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->zero_points();
}

int64_t q_per_channel_axis(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->axis();
}

Tensor make_per_channel_quantized_tensor_cpu(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  Tensor dst = at::_empty_per_channel_affine_quantized(
      self.sizes(),
      scales,
      zero_points,
      axis,
      self.options().dtype(toQIntType(self.scalar_type())));
  Tensor self_contig = self.contiguous();
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "per_channel_affine_qtensor", [&]() {
        underlying_t* self_data = self_contig.data_ptr<underlying_t>();
        underlying_t* dst_data =
            reinterpret_cast<underlying_t*>(dst.data_ptr<scalar_t>());
        if (self.numel() > 0) {
          memcpy(dst_data, self_data, self.nbytes());
        }
      });
  return dst;
}

Tensor& set_storage_quantized_(
    Tensor& self,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef sizes,
    IntArrayRef strides) {
  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_storage_keep_dtype(std::move(storage));
  self_->set_storage_offset(storage_offset);
  self_->set_sizes_and_strides(sizes, strides);
  return self;
}

QScheme qscheme_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  return quantizer->qscheme();
}

Tensor quantized_clone(
    const Tensor& self,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);

  // TODO: To support all features of MemoryFormat::Preserve we need to add
  // _empty_affine_quantized_strided function and use it similarly to
  // Tensor clone(const Tensor& src, std::optional<c10::MemoryFormat>
  // optional_memory_format) if (self.is_non_overlapping_and_dense()) ->
  // _empty_affine_quantized_strided
  if (memory_format == MemoryFormat::Preserve) {
    memory_format = self.suggest_memory_format();
  }

  Tensor dst;
  if (self.qscheme() == at::kPerTensorAffine) {
    dst = at::_empty_affine_quantized(
        self.sizes(),
        self.options().memory_format(memory_format),
        self.q_scale(),
        self.q_zero_point(),
        std::nullopt);
  } else if (self.qscheme() == at::kPerChannelAffine) {
    dst = at::_empty_per_channel_affine_quantized(
        self.sizes(),
        self.q_per_channel_scales(),
        self.q_per_channel_zero_points(),
        self.q_per_channel_axis(),
        self.options().memory_format(memory_format),
        std::nullopt);
  } else {
    TORCH_CHECK(false, "clone for quantized Tensor only works for \
      PerTensorAffine and PerChannelAffine qscheme right now");
  }

  at::native::copy_(dst, self, false);

  return dst;
}

bool equal_quantized_cpu(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.device().type() == kCPU && other.device().type() == kCPU,
      "quantized_equal is implemented only for the QuantizedCPU backend");
  if (!self.is_quantized() || !other.is_quantized()) {
    return false;
  }

  // Delegate to virtual equalTo method. This will ensure different concrete
  // Quantizers can have specific logic for comparison
  auto self_quantizer = get_qtensorimpl(self)->quantizer();
  auto other_quantizer = get_qtensorimpl(other)->quantizer();
  if (!self_quantizer->equalTo(other_quantizer)) {
    return false;
  }

  // Sizes and element types must be the same
  if (self.sizes() != other.sizes()) {
    return false;
  }
  if (self.scalar_type() != other.scalar_type()) {
    return false;
  }

  // Data must be the same
  auto self_contig = self.contiguous();
  auto other_contig = other.contiguous();

  void* self_data = self_contig.data_ptr();
  void* other_data = other_contig.data_ptr();
  auto data_size = self.numel() * self.element_size();
  // For QUint4x2 and QUInt2x4, two elements are packed in one byte
  if (self.scalar_type() == kQUInt4x2 || self.scalar_type() == kQUInt2x4) {
      TORCH_INTERNAL_ASSERT(self.element_size() == 1);
      data_size = (data_size>>1) + (data_size&1);
  }
  return 0 == memcmp(self_data, other_data, data_size);
}

/* Calculate the quantization params for the activation tensor */
std::tuple<double, int64_t> _choose_qparams_per_tensor(
    const Tensor& self,
    bool reduce_range) {
  at::Tensor a;
  auto input_contig = self.contiguous();
  float x_min = input_contig.min().item<float>();
  float x_max = input_contig.max().item<float>();

  if (reduce_range && at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    reduce_range = false;
  }

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/255,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  return std::make_tuple(q_params.scale, q_params.zero_point);
}

static float calculate_quant_loss(
    const float* input,
    int64_t numel,
    float xmin,
    float xmax,
    float* q_input,
    int64_t bit_width) {
  xmin = static_cast<at::Half>(xmin);
  float data_range = xmax - xmin;
  float qmax = static_cast<float>((1 << bit_width) - 1);
  float scale = data_range == 0
      ? 1.0f
      : static_cast<float>(static_cast<at::Half>(data_range / qmax));
  float inverse_scale = scale == 0 ? 1.0f : 1.0f / scale;

  float norm = 0.0f;
  int i = 0;

  // TODO add FBGEMM kernel
  // #ifdef USE_FBGEMM
  // #endif

  // remainder loop
  for (; i < numel; i++) {
    q_input[i] = std::max(
        0.0f, std::min<float>(std::nearbyint((input[i] - xmin) * inverse_scale), qmax));
    q_input[i] = q_input[i] * scale + xmin;
    norm += (input[i] - q_input[i]) * (input[i] - q_input[i]);
  }
  return std::sqrt(norm);
}

/*
  Helper function to find the best min/max for a tensor to calculate qparams.
  It uses a greedy approach to nudge the min and max and calculate the l2 norm
  and tries to minimize the quant error by doing `torch.norm(x-fake_quant(x,s,z))`
  Returns the optimized xmax and xmin value of the tensor.
*/
std::tuple<Tensor, Tensor> choose_qparams_optimized(
    const at::Tensor& input_tensor,
    int64_t numel,
    const int64_t n_bins,
    const double ratio,
    int64_t bit_width) {
  const float* input_row = input_tensor.const_data_ptr<float>();
  TORCH_CHECK_VALUE(input_row != nullptr, "input tensor is empty and has no data");

  if (numel < 0 || numel > input_tensor.numel()) {
    TORCH_CHECK(false, "numel is out of the bound of input tensor");
  }

  TORCH_CHECK(numel <= input_tensor.numel(), "numel ", numel,
      " greater than input_tensor.numel() ", input_tensor.numel());

  float xmin = *std::min_element(input_row, input_row + numel);
  float xmax = *std::max_element(input_row, input_row + numel);
  float n_bins_float = static_cast<float>(n_bins);

  float stepsize = (xmax - xmin) / n_bins_float;
  float min_bins = static_cast<float>(n_bins_float* (1.0 - ratio));
  Tensor input_tensor_contig = input_tensor.contiguous();
  const float* input = input_tensor_contig.const_data_ptr<float>();
  std::vector<float> q_input(numel);

  float loss =
      calculate_quant_loss(input, numel, xmin, xmax, q_input.data(), bit_width);
  float best_loss = loss;

  float cur_min = xmin;
  float cur_max = xmax;
  float cur_loss = loss;

  float thr = min_bins * stepsize;
  while (cur_min + thr < cur_max) {
    // move left
    float loss1 = calculate_quant_loss(
        input, numel, cur_min + stepsize, cur_max, q_input.data(), bit_width);
    // move right
    float loss2 = calculate_quant_loss(
        input, numel, cur_min, cur_max - stepsize, q_input.data(), bit_width);
    if (cur_loss < loss1 && cur_loss < loss2 && cur_loss < best_loss) {
      // found a local optima
      best_loss = cur_loss;
      xmin = cur_min;
      xmax = cur_max;
    }
    if (loss1 < loss2) {
      cur_min = cur_min + stepsize;
      cur_loss = loss1;
    } else {
      cur_max = cur_max - stepsize;
      cur_loss = loss2;
    }
  }

  at::Tensor xmax_tensor = at::empty({1});
  at::Tensor xmin_tensor = at::empty({1});
  xmax_tensor[0] = xmax;
  xmin_tensor[0] = xmin;
  return std::make_tuple(xmax_tensor, xmin_tensor);
}
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 23 function(s).

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
- `ATen/NativeFunctions.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/quantized/cpu/QuantUtils.h`
- `ATen/quantized/QTensorImpl.h`
- `ATen/quantized/Quantizer.h`
- `c10/util/irange.h`
- `cmath`
- `utility`


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

- **File Documentation**: `QTensor.cpp_docs.md`
- **Keyword Index**: `QTensor.cpp_kw.md`
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
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`PackedParams.h_docs.md_docs.md`](./PackedParams.h_docs.md_docs.md)
- [`AffineQuantizerBase.cpp_docs.md_docs.md`](./AffineQuantizerBase.cpp_docs.md_docs.md)
- [`FakeQuantAffine.h_kw.md_docs.md`](./FakeQuantAffine.h_kw.md_docs.md)
- [`FakeQuantPerChannelAffine.cpp_kw.md_docs.md`](./FakeQuantPerChannelAffine.cpp_kw.md_docs.md)
- [`TensorCompare.cpp_kw.md_docs.md`](./TensorCompare.cpp_kw.md_docs.md)
- [`Copy.h_kw.md_docs.md`](./Copy.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `QTensor.cpp_docs.md_docs.md`
- **Keyword Index**: `QTensor.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
