# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qsoftmax.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qsoftmax.cpp_docs.md`
- **Size**: 7,361 bytes (7.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qsoftmax.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qsoftmax.cpp`
- **Size**: 4,689 bytes (4.58 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <torch/library.h>

#ifdef USE_PYTORCH_QNNPACK
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <pytorch_qnnpack.h>

#include <utility>
#endif // USE_PYTORCH_QNNPACK

namespace at::native {

namespace {

#ifdef USE_PYTORCH_QNNPACK

constexpr static float qnnpack_softmax_output_scale = 0x1.0p-8f;
constexpr static int qnnpack_softmax_output_zero_point = 0;

bool is_qnnpack_compatible(
    const Tensor& qx,
    const double output_scale,
    const int64_t output_zero_point) {
  return (
      (qx.qscheme() == kPerTensorAffine ||
       qx.qscheme() == kPerTensorSymmetric) &&
      qx.scalar_type() == c10::kQUInt8 && qx.ndimension() > 0 &&
      output_scale == qnnpack_softmax_output_scale &&
      output_zero_point == qnnpack_softmax_output_zero_point);
}

Tensor qsoftmax_qnnpack(const Tensor& qx, const int64_t dim) {
  /*
    Cases for contiguity/dimensionality
    1) stride along target dim is 1
        requires no change to qx
    2) dim is the last dimension (but qx is not contiguous)
        requires using qx.contiguous()
    3) other
        requires permuting qx.contiguous()
   */

  const int64_t last_dim = qx.dim() - 1;
  std::optional<std::vector<int64_t>> permuted_dims = std::nullopt;
  std::optional<at::Tensor> qx_contig = std::nullopt;
  const at::Tensor* qx_contig_ptr = nullptr;

  if (qx.stride(dim) == 1) {
    qx_contig_ptr = &qx;
  } else if (dim == last_dim) {
    qx_contig = qx.contiguous();
    qx_contig_ptr = &qx_contig.value();
  } else {
    permuted_dims = std::vector<int64_t>(qx.dim());
    std::iota(permuted_dims->begin(), permuted_dims->end(), 0);
    permuted_dims->at(last_dim) = dim;
    permuted_dims->at(dim) = last_dim;
    qx_contig = qx.permute(permuted_dims.value()).contiguous();
    qx_contig_ptr = &qx_contig.value();
  }

  at::Tensor qy = at::_empty_affine_quantized(
      qx_contig_ptr->sizes(),
      at::device(kCPU)
          .dtype(qx.scalar_type())
          .memory_format(qx_contig_ptr->suggest_memory_format()),
      qnnpack_softmax_output_scale,
      qnnpack_softmax_output_zero_point,
      std::nullopt);

  const size_t channels = qx.size(dim);
  const float input_scale = static_cast<float>(qx.q_scale());
  const uint32_t flags = 0;
  const size_t batch_size = qx.numel() / channels;
  const uint8_t* input =
      reinterpret_cast<const uint8_t*>(qx_contig_ptr->data_ptr<c10::quint8>());
  const size_t input_stride = channels;
  uint8_t* output = reinterpret_cast<uint8_t*>(qy.data_ptr<c10::quint8>());
  const size_t output_stride = channels;

  initQNNPACK();
  pytorch_qnnp_operator_t softargmax = nullptr;

  pytorch_qnnp_status status = pytorch_qnnp_create_softargmax_nc_q8(
      channels,
      input_scale,
      qnnpack_softmax_output_zero_point,
      qnnpack_softmax_output_scale,
      flags,
      &softargmax);
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,
      "failed to create QNNPACK Softmax operator");
  TORCH_CHECK_NOTNULL(softargmax);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter> softmax_op(
    softargmax);

  status = pytorch_qnnp_setup_softargmax_nc_q8(
      softargmax, batch_size, input, input_stride, output, output_stride);
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Softmax operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();
  status = pytorch_qnnp_run_operator(softargmax, threadpool);
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,
      "failed to run QNNPACK Softmax operator");

  return permuted_dims.has_value() ? qy.permute(permuted_dims.value()) : std::move(qy);
}

#endif // USE_PYTORCH_QNNPACK

Tensor qsoftmax_naive(
    const Tensor& qx,
    const int64_t dim,
    const double output_scale,
    const int64_t output_zero_point) {
  Tensor rx = at::dequantize(qx);
  Tensor ry = at::softmax(rx, dim);
  return at::quantize_per_tensor(
      ry, output_scale, output_zero_point, qx.scalar_type());
}

Tensor qsoftmax(
    const Tensor& qx,
    const int64_t dim,
    const double output_scale,
    const int64_t output_zero_point) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      is_qnnpack_compatible(qx, output_scale, output_zero_point)) {
    return qsoftmax_qnnpack(qx, dim);
  }
#endif // USE_PYTORCH_QNNPACK
  return qsoftmax_naive(qx, dim, output_scale, output_zero_point);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::softmax"), TORCH_FN(qsoftmax));
}

} // namespace

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

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

- `ATen/ATen.h`
- `torch/library.h`
- `ATen/native/quantized/cpu/init_qnnpack.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `pytorch_qnnpack.h`
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

- **File Documentation**: `qsoftmax.cpp_docs.md`
- **Keyword Index**: `qsoftmax.cpp_kw.md`
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

- **File Documentation**: `qsoftmax.cpp_docs.md_docs.md`
- **Keyword Index**: `qsoftmax.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
