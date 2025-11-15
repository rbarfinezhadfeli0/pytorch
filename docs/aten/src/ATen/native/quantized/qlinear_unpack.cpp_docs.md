# Documentation: `aten/src/ATen/native/quantized/qlinear_unpack.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/qlinear_unpack.cpp`
- **Size**: 2,876 bytes (2.81 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
/*
The dispatch registrations at the end of this file applies to fbgemm, qnnpack, and cudnn backends.
The correct unpack backend function is determined using runtime polymorphism through the packed_weight pointer,
which is of type intrusive_ptr<LinearPackedParamsBase> and points to either a PackedLinearWeightsQnnp,
PackedLinearWeights (Fbgemm), or PackedLinearWeightsCudnn at runtime, which all inherit from LinearPackedParamsBase.
The implementations for the unpack functions can be found in /cpu/LinearUnpackImpl.cpp, for fbgemm&qnnpack
and /cudnn/linear_unpack_impl.cpp, for cudnn.
*/
#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/library.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/custom_class.h>
#include <torch/library.h>

namespace at::native {
namespace {

class QLinearUnpackWeightInt8 final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

class QLinearUnpackWeightFp16 final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    auto& ctx = at::globalContext();

    TORCH_CHECK(
        ctx.qEngine() != at::QEngine::QNNPACK,
        "quantized::linear_unpack_fp16 is currently "
        "not supported by QNNPACK");

    return packed_weight->unpack();
  }
};

class QLinearUnpackWeightInt8Legacy final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const at::Tensor& packed_weight) {
    TORCH_CHECK(false,
        "quantized.linear_unpack(Tensor) is unsupported! Please "
        "upgrade your model to use the newer quantized.linear_"
        "unpack(LinearPackedParamsBase) overload");
  }
};

class QLinearUnpackWeightFp16Legacy final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const at::Tensor& packed_weight) {
    TORCH_CHECK(false,
        "quantized.linear_unpack(Tensor) is unsupported! Please "
        "upgrade your model to use the newer quantized.linear_"
        "unpack(LinearPackedParamsBase) overload");
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack.legacy"), TORCH_FN(QLinearUnpackWeightInt8Legacy::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16.legacy"), TORCH_FN(QLinearUnpackWeightFp16Legacy::run));
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack"), TORCH_FN(QLinearUnpackWeightInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16"), TORCH_FN(QLinearUnpackWeightFp16::run));
}

} // namespace
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `QLinearUnpackWeightInt8`, `QLinearUnpackWeightFp16`, `QLinearUnpackWeightInt8Legacy`, `QLinearUnpackWeightFp16Legacy`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/native/quantized/cpu/fbgemm_utils.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `ATen/native/quantized/library.h`
- `ATen/native/quantized/PackedParams.h`
- `torch/custom_class.h`
- `torch/library.h`


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
- [`TensorFactories.cpp_docs.md`](./TensorFactories.cpp_docs.md)


## Cross-References

- **File Documentation**: `qlinear_unpack.cpp_docs.md`
- **Keyword Index**: `qlinear_unpack.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
