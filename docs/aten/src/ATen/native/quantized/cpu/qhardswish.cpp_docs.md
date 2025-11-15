# Documentation: `aten/src/ATen/native/quantized/cpu/qhardswish.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qhardswish.cpp`
- **Size**: 3,373 bytes (3.29 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#endif

#include <algorithm>

namespace at::native {

DEFINE_DISPATCH(qhardswish_stub);

namespace {

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_hardswish(const Tensor& qx, Tensor& qy) {
  TORCH_CHECK(qx.ndimension() > 0, "qnnpack_hardswish(): Got empty input tensor");
  TORCH_CHECK(qx.scalar_type() == c10::kQUInt8,
                "qnnpack_hardswish(): Expected input data type to be ",
                toString(c10::kQUInt8),
                " but got ",
                toString(qx.scalar_type()));
  initQNNPACK();

  size_t num_elems = qx.numel() / qx.size(0);
  const auto i_zero_point = qx.q_zero_point();
  const auto i_scale = qx.q_scale();
  const auto o_zero_point = qy.q_zero_point();
  const auto o_scale = qy.q_scale();

  pytorch_qnnp_operator_t hardswish_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_hardswish_nc_q8(
    num_elems, // channels
    i_zero_point,
    i_scale,
    o_zero_point,
    o_scale,
    std::numeric_limits<uint8_t>::min(), // output min
    std::numeric_limits<uint8_t>::max(), // output max
    0, // flags
    &hardswish_op);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(hardswish_op);

  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK Hardswish operator");

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_hardswish_nc_q8(
    hardswish_op,
    qx.size(0), // batch size
    (uint8_t*)qx.data_ptr<c10::quint8>(), // input data
    num_elems, // input stride
    (uint8_t*)qy.data_ptr<c10::quint8>(), // output data
    num_elems); // output stride
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK Hardswish operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(hardswish_op, threadpool);

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK Hardswish operator");
  return qy;
}
#endif // USE_PYTORCH_QNNPACK

} // namespace

static Tensor quantized_hardswish(const Tensor& qx, double output_scale, int64_t output_zero_point) {
  Tensor qy = at::_empty_affine_quantized(
      qx.sizes(),
      at::device(kCPU).dtype(qx.scalar_type()),
      output_scale,
      output_zero_point,
      qx.suggest_memory_format());
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    Tensor qx_contig = qx.contiguous(qx.suggest_memory_format());
    qnnpack_hardswish(qx_contig, qy);
    return qy;
  }
#endif  // USE_PYTORCH_QNNPACK
  qhardswish_stub(qx.device().type(), qx, qy);
  return qy;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::hardswish"), TORCH_FN(quantized_hardswish));
}

}  // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `static`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Context.h`
- `torch/library.h`
- `ATen/native/quantized/cpu/QuantizedOps.h`
- `ATen/native/quantized/cpu/init_qnnpack.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `ATen/Functions.h`
- `ATen/ops/_empty_affine_quantized.h`
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

- **File Documentation**: `qhardswish.cpp_docs.md`
- **Keyword Index**: `qhardswish.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
