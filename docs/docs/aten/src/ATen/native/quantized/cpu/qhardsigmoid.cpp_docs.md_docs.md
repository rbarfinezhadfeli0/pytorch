# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qhardsigmoid.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qhardsigmoid.cpp_docs.md`
- **Size**: 6,576 bytes (6.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qhardsigmoid.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qhardsigmoid.cpp`
- **Size**: 3,726 bytes (3.64 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/hardsigmoid_native.h>
#endif

#include <algorithm>

namespace at::native {

DEFINE_DISPATCH(qhardsigmoid_stub);

namespace {

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_hardsigmoid(Tensor input) {
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_hardsigmoid(): Got empty input tensor");
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
                "qnnpack_hardsigmoid(): Expected input data type ",
                toString(c10::kQUInt8),
                " but got ",
                toString(input.scalar_type()));
  initQNNPACK();

  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  size_t num_elems = input_contig.numel() / input_contig.size(0);
  const auto i_zero_point = input_contig.q_zero_point();
  const auto i_scale = input_contig.q_scale();
  constexpr float o_scale = 1.0f / 256.0f;
  constexpr int32_t o_zero_point = 0;

  pytorch_qnnp_operator_t hardsigmoid_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_hardsigmoid_nc_q8(
    num_elems, // channels
    i_zero_point,
    i_scale,
    o_zero_point,
    o_scale,
    std::numeric_limits<uint8_t>::min(), // output min
    std::numeric_limits<uint8_t>::max(), // output max
    0, // flags
    &hardsigmoid_op);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(hardsigmoid_op);

  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK Hardsigmoid operator");
  Tensor qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    at::device(kCPU).dtype(input_contig.dtype()),
    o_scale,
    o_zero_point,
    input_contig.suggest_memory_format());

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_hardsigmoid_nc_q8(
    hardsigmoid_op,
    input_contig.size(0), // batch size
    (uint8_t*)input_contig.data_ptr<c10::quint8>(), // input data
    num_elems, // input stride
    (uint8_t*)qy.data_ptr<c10::quint8>(), // output data
    num_elems); // output stride
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK Hardsigmoid operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(hardsigmoid_op, threadpool);

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK Hardsigmoid operator");
  return qy;
}
#endif // USE_PYTORCH_QNNPACK

} // namespace
Tensor hardsigmoid_quantized_cpu(const Tensor& qx) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    return qnnpack_hardsigmoid(qx);
  }
#endif  // USE_PYTORCH_QNNPACK
  Tensor qy;
  qhardsigmoid_stub(qx.device().type(), qx, qy);
  return qy;
}

Tensor& hardsigmoid_out_quantized_cpu(const Tensor& qx, Tensor& result) {
  // Note: we create a new temporary tensor because the output of hardsigmoid
  // usually has different quantization parameters from the input, and
  // quantization are currently only supported per entire tensor or per entire
  // channel of a tensor.
  Tensor qy = hardsigmoid_quantized_cpu(qx);
  result.copy_(qy);
  return result;
}

}  // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Context.h`
- `ATen/native/quantized/cpu/QuantizedOps.h`
- `ATen/native/quantized/cpu/init_qnnpack.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_empty_affine_quantized.h`
- `ATen/ops/hardsigmoid_native.h`
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

- **File Documentation**: `qhardsigmoid.cpp_docs.md`
- **Keyword Index**: `qhardsigmoid.cpp_kw.md`
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

- **File Documentation**: `qhardsigmoid.cpp_docs.md_docs.md`
- **Keyword Index**: `qhardsigmoid.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
