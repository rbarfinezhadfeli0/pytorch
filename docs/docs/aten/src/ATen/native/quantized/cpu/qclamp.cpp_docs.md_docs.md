# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qclamp.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qclamp.cpp_docs.md`
- **Size**: 7,919 bytes (7.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qclamp.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qclamp.cpp`
- **Size**: 4,967 bytes (4.85 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/clamp_native.h>
#include <ATen/ops/hardtanh_native.h>
#endif

#include <algorithm>

namespace at::native {

DEFINE_DISPATCH(qclamp_stub);
DEFINE_DISPATCH(qclamp_min_stub);
DEFINE_DISPATCH(qclamp_max_stub);

namespace {

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_clamp(Tensor input, const Scalar& min, const Scalar& max) {

  TORCH_CHECK(input.ndimension() > 0, "qnnpack_clamp(): Got empty input tensor");

  initQNNPACK();

  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }

  auto min_f = min.to<float>();
  auto max_f = max.to<float>();
  uint8_t min_q =
      at::native::quantize_val<quint8>(input.q_scale(), input.q_zero_point(), min_f).val_;
  uint8_t max_q =
      at::native::quantize_val<quint8>(input.q_scale(), input.q_zero_point(), max_f).val_;

  pytorch_qnnp_operator_t clamp_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_clamp_nc_u8(
    num_elems, // channels
    min_q,
    max_q,
    0, // flags
    &clamp_op);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(clamp_op);

  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK Clamp operator");

  Tensor qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    input_contig.options(),
    input_contig.q_scale(),
    input_contig.q_zero_point());

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_clamp_nc_u8(
    clamp_op,
    input_contig.size(0), // batch_size
    (uint8_t*)input_contig.data_ptr<c10::quint8>(), // input_data
    num_elems, // input_stride
    (uint8_t*)qy.data_ptr<c10::quint8>(), // output_data
    num_elems); // output_stride
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK Clamp operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(clamp_op, threadpool);

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK Clamp operator");
  return qy;
}

#endif // USE_PYTORCH_QNNPACK

Tensor quantized_clamp_impl(
    const Tensor& qx,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  Tensor qy;
  if (min && max) {
#ifdef USE_PYTORCH_QNNPACK
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
        qx.scalar_type() == kQUInt8) {
      return qnnpack_clamp(qx, *min, *max);
    }
#endif
    qclamp_stub(qx.device().type(), qx, *min, *max, qy);
  } else {
#ifdef USE_PYTORCH_QNNPACK
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          false, "Both min and max should be specified for quantized clamp!");
    }
#endif
    if (max) {
      qclamp_max_stub(qx.device().type(), qx, *max, qy);
    } else if (min) {
      qclamp_min_stub(qx.device().type(), qx, *min, qy);
    } else {
      TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
    }
  }
  return qy;
}
} // namespace

// at::native functions for the native_functions.yaml
Tensor clamp_quantized_cpu(
    const Tensor& qx,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
    qy = quantized_clamp_impl(qx, min, max);
  });
  return qy;
}

// hardtanh is clamp with default min==-1.0f and default max==1.0f
Tensor hardtanh_quantized_cpu(
    const Tensor& qx,
    const Scalar& min,
    const Scalar& max) {
  Tensor qy;
  qy = quantized_clamp_impl(qx, min, max);
  return qy;
}

Tensor& hardtanh_out_quantized_cpu(const Tensor& qx,
    const Scalar& min,
    const Scalar& max,
    Tensor& result) {
  result = quantized_clamp_impl(qx, min, max);
  return result;
}

Tensor& hardtanh_quantized_cpu_(
    Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  Tensor qy;
  qy = quantized_clamp_impl(self, min, max);
  // This can be optimized in a future PR if it becomes a bottleneck.
  self.copy_(qy);
  return self;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::clamp"), TORCH_FN(clamp_quantized_cpu));
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

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Context.h`
- `ATen/Dispatch.h`
- `torch/library.h`
- `ATen/native/quantized/AffineQuantizerBase.h`
- `ATen/native/quantized/cpu/QuantizedOps.h`
- `ATen/native/quantized/cpu/init_qnnpack.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `c10/util/irange.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_empty_affine_quantized.h`
- `ATen/ops/clamp_native.h`
- `ATen/ops/hardtanh_native.h`
- `algorithm`


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

- **File Documentation**: `qclamp.cpp_docs.md`
- **Keyword Index**: `qclamp.cpp_kw.md`
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
- [`qembeddingbag_prepack.h_docs.md_docs.md`](./qembeddingbag_prepack.h_docs.md_docs.md)
- [`qdropout.cpp_docs.md_docs.md`](./qdropout.cpp_docs.md_docs.md)
- [`qelu.cpp_docs.md_docs.md`](./qelu.cpp_docs.md_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md_docs.md`](./qembeddingbag_unpack.cpp_docs.md_docs.md)
- [`LinearUnpackImpl.cpp_kw.md_docs.md`](./LinearUnpackImpl.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `qclamp.cpp_docs.md_docs.md`
- **Keyword Index**: `qclamp.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
