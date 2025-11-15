# Documentation: `aten/src/ATen/native/quantized/cpu/ChannelShuffle.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/ChannelShuffle.cpp`
- **Size**: 3,850 bytes (3.76 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/channel_shuffle_native.h>
#endif

namespace at::native {

#ifdef USE_PYTORCH_QNNPACK
namespace {
Tensor quantized_channel_shuffle_impl(
    const Tensor& self,
    int64_t groups) {

  TORCH_CHECK(
      groups > 0,
      "Number of groups to divide channels in must be positive.",
      " Value of groups:", groups);
  TORCH_CHECK(
      self.dim() == 4,
      "channel_shuffle expects 4D input, but got input with sizes ",
      self.sizes());
  TORCH_CHECK(
      self.scalar_type() == kQUInt8,
      "Quantized channel shuffle works only on ",
      toString(c10::kQUInt8),
      " but got ", self.scalar_type());
  const Tensor self_nhwc = self.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::native::empty_affine_quantized(
      self_nhwc.sizes(),
      kQUInt8,
      std::nullopt /* layout */,
      kCPU,
      std::nullopt /* pin_memory */,
      self_nhwc.q_scale(),
      self_nhwc.q_zero_point(),
      MemoryFormat::ChannelsLast);

  // Degenerate case of just copying.
  if (groups == 1) {
    qy.copy_(self_nhwc);
    return qy.contiguous(self.suggest_memory_format());
  }

  int64_t channels = self.size(1);
  TORCH_CHECK(channels > 0,
             "Number of channels must be positive, got:", channels);
  TORCH_CHECK((channels % groups) == 0,
             "Number of channels must be divisible gy groups. Got ",
             channels, " channels and ", groups, " groups.");

  initQNNPACK();

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_channel_shuffle_nc_x8(
      groups /* groups */,
      channels / groups /* group channels */,
      0 /* flags */,
      &qnnpack_operator);
  TORCH_INTERNAL_ASSERT(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK ChannelShuffle operator");

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_channel_shuffle_nc_x8(
      qnnpack_uniq_ptr.get(),
      self_nhwc.numel() / channels /* batch size */,
      (uint8_t*)self_nhwc.data_ptr<c10::quint8>() /* self data */,
      channels /* self stride */,
      (uint8_t*)qy.data_ptr<c10::quint8>() /* qy data */,
      channels /* qy stride */);
  TORCH_INTERNAL_ASSERT(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK ChannelShuffle operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK ChannelShuffle operator");

  return qy.contiguous(self.suggest_memory_format());
}
} // namespace
#endif

// at::native functions for the native_functions.yaml
Tensor channel_shuffle_quantized_cpu(
    const Tensor& self,
    int64_t groups) {
#ifdef USE_PYTORCH_QNNPACK
  return quantized_channel_shuffle_impl(self, groups);
#endif
  // If QNNPACK is not available then fall back to the
  // non quantized path.
  return at::native::channel_shuffle(self, groups);
}

// Keep the registry in the anonymous namespace.
namespace {
class QChannelShuffle final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, int64_t groups) {
    return channel_shuffle_quantized_cpu(qx, groups);
  }
};

} // namespace

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `QChannelShuffle`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/core/boxing/KernelFunction.h`
- `ATen/native/quantized/cpu/init_qnnpack.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_empty_affine_quantized_native.h`
- `ATen/ops/channel_shuffle_native.h`


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

- **File Documentation**: `ChannelShuffle.cpp_docs.md`
- **Keyword Index**: `ChannelShuffle.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
