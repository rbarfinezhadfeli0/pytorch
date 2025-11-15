# Documentation: `docs/aten/src/ATen/native/quantized/cpu/ReduceOps.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/ReduceOps.cpp_docs.md`
- **Size**: 10,538 bytes (10.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/ReduceOps.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/ReduceOps.cpp`
- **Size**: 7,572 bytes (7.39 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>         // for _empty_affine_q...
#include <ATen/ops/mean.h>                            // for mean
#include <ATen/ops/mean_native.h>                     // for mean_out_quanti...
#include <ATen/ops/quantize_per_tensor.h>             // for quantize_per_te...
#include <ATen/ops/std.h>
#include <ATen/ops/std_native.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

namespace at::native {

DEFINE_DISPATCH(qmean_inner_dim_stub);
DEFINE_DISPATCH(qstd_inner_dim_stub);

// If mean/std is taken in the innermost dims, the fast path can be used.
static inline bool is_innnermost_dim(
    const Tensor& self,
    OptionalIntArrayRef opt_dim) {
  if (!opt_dim.has_value()) {
    return true;
  }
  auto dims = opt_dim.value().vec();
  auto ndim = self.dim();
  maybe_wrap_dims(dims, ndim);
  std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
  bool is_innermost = dims.empty() || dims[0] == ndim - 1;
  for (size_t i = 1; i < dims.size(); ++i) {
    is_innermost = is_innermost && (dims[i] == dims[i-1] - 1);
  }
  return is_innermost;
}

static inline bool is_mean_inner_dim_fast_path(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    std::optional<ScalarType> opt_dtype) {
  bool is_fast_path =
      is_innnermost_dim(self, opt_dim) &&
      (!opt_dtype.has_value() || opt_dtype.value() == self.scalar_type());
  return is_fast_path;
}

#ifdef USE_PYTORCH_QNNPACK
static Tensor qnnpack_mean(const Tensor& input, IntArrayRef dim, bool keepdim) {
  Tensor output;
  TORCH_CHECK(
      input.ndimension() == 4,
      "qnnpack_global_average_pool: Expected input to be 4-dimensional: got ",
      input.ndimension());
  TORCH_CHECK(
      dim.size() == 2,
      "qnnpack_global_average_pool: dim size must be a tuple of two ints");
  TORCH_CHECK(
      dim[0] == 2 && dim[1] == 3,
      "qnnpack_global_average_pool: Reduction dimensions must match last 2 dimensions of input tensor")

  const int64_t batch_size = input.size(0);
  const int64_t inC = input.size(1);
  const int64_t inH = input.size(2);
  const int64_t inW = input.size(3);

  Tensor input_contig = input.contiguous(MemoryFormat::ChannelsLast);

  initQNNPACK();
  const auto scale = input_contig.q_scale();
  const auto zero_point = input_contig.q_zero_point();
  const auto outC = inC;

  output = at::_empty_affine_quantized(
      keepdim ? IntArrayRef{batch_size, outC, 1, 1}
              : IntArrayRef{batch_size, outC},
      at::device(kCPU).dtype(kQUInt8),
      scale,
      zero_point);

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};
  const pytorch_qnnp_status createStatus =
      pytorch_qnnp_create_global_average_pooling_nwc_q8(
          inC,
          zero_point,
          scale,
          zero_point,
          scale,
          std::numeric_limits<uint8_t>::min() /* output min */,
          std::numeric_limits<uint8_t>::max() /* output max */,
          0,
          &qnnpack_operator);

  CAFFE_ENFORCE(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK Global Average Pooling operator");
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  const pytorch_qnnp_status setupStatus =
      pytorch_qnnp_setup_global_average_pooling_nwc_q8(
          qnnpack_operator,
          batch_size,
          inH * inW,
          (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
          inC,
          (uint8_t*)output.data_ptr<c10::quint8>() /* output data */,
          outC);
  CAFFE_ENFORCE(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Global Average Pooling operator");
  pthreadpool_t threadpool = caffe2::pthreadpool_();
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Global Average Pool operator");
  return output;
}
#endif
Tensor& mean_out_quantized_cpu(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    std::optional<ScalarType> opt_dtype,
    Tensor& result) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      self.scalar_type() == kQUInt8 && opt_dim.has_value()) {
    auto dim = opt_dim.value();
    // QNNPACK currently is only supported for NCHW + dim=(2, 3)
    // Remove these checks after generic version is implemented.
    if (self.ndimension() == 4 && dim.size() == 2 && dim[0] == 2 && dim[1] == 3) {
      result = qnnpack_mean(self, dim, keepdim);
      return result;
    }
  }
#endif

  // Take average in the innermost dimensions
  if (self.is_contiguous(c10::MemoryFormat::Contiguous) &&
      is_mean_inner_dim_fast_path(self, opt_dim, opt_dtype)) {
    qmean_inner_dim_stub(self.device().type(), self, opt_dim, keepdim, opt_dtype, result);
    return result;
  }
  auto self_dequantized = self.dequantize();
  auto result_dequantized = at::mean(self_dequantized, opt_dim, keepdim, opt_dtype);
  result = at::quantize_per_tensor(
      result_dequantized,
      self.q_scale(),
      self.q_zero_point(),
      opt_dtype.value_or(self.scalar_type()));
  return result;
}

Tensor mean_quantized_cpu(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  Tensor result;
  mean_out_quantized_cpu(self, opt_dim, keepdim, dtype, result);
  return result;
}

// qstd
static inline bool is_std_inner_dim_fast_path(
    const Tensor& self,
    OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction) {
  // Do not enter fast path if there are too few elements
  IntArrayRef dims = dim.has_value() ? dim.value() : IntArrayRef();
  auto all_dims = std::vector<int64_t>(self.dim());
  std::iota(all_dims.begin(), all_dims.end(), 0);
  dims = dims.empty() ? all_dims : dims;
  bool has_correction = !correction.value_or(1).equal(0);
  int64_t num_ele = 1;
  for (auto d : dims) {
    num_ele *= self.size(d);
  }
  if (num_ele == 1 && has_correction) {
    return false;
  }
  return is_innnermost_dim(self, dims);
}

Tensor& std_out_quantized_cpu(
    const Tensor& self,
    OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result) {
  // Fast path
  if (self.is_contiguous(c10::MemoryFormat::Contiguous) &&
      is_std_inner_dim_fast_path(self, dim, correction)) {
    qstd_inner_dim_stub(self.device().type(), self, dim, correction, keepdim, result);
    return result;
  }

  // Reference path
  auto self_dequantized = self.dequantize();
  auto result_dequantized = at::std(self_dequantized, dim, correction, keepdim);
  result = at::quantize_per_tensor(
      result_dequantized,
      self.q_scale(),
      self.q_zero_point(),
      self.scalar_type());
  return result;
}

Tensor std_quantized_cpu(
    const Tensor& self,
    OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result;
  std_out_quantized_cpu(self, dim, correction, keepdim, result);
  return result;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Context.h`
- `ATen/NamedTensorUtils.h`
- `ATen/native/quantized/cpu/init_qnnpack.h`
- `ATen/native/quantized/cpu/QuantizedOps.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_empty_affine_quantized.h`
- `ATen/ops/mean.h`
- `ATen/ops/mean_native.h`
- `ATen/ops/quantize_per_tensor.h`
- `ATen/ops/std.h`
- `ATen/ops/std_native.h`
- `ATen/ops/zeros_like_ops.h`


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

- **File Documentation**: `ReduceOps.cpp_docs.md`
- **Keyword Index**: `ReduceOps.cpp_kw.md`
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

- **File Documentation**: `ReduceOps.cpp_docs.md_docs.md`
- **Keyword Index**: `ReduceOps.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
