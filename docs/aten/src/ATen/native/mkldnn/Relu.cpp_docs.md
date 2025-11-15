# Documentation: `aten/src/ATen/native/mkldnn/Relu.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/Relu.cpp`
- **Size**: 2,577 bytes (2.52 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/relu_native.h>                // for mkldnn_relu, mkldnn_...
#include <ATen/ops/threshold_backward_native.h>  // for mkldnn_relu_backward
#endif

#if !AT_MKLDNN_ENABLED()

namespace at::native {

Tensor mkldnn_relu(const Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_relu_(Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu_: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_relu_backward(const Tensor& grad_output, const Tensor& input, const Scalar& threshold) {
  TORCH_CHECK(false, "mkldnn_relu_backward: ATen not compiled with MKLDNN support");
}

}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at::native {

Tensor mkldnn_relu(const Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  const ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor& mkldnn_relu_(Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu_: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return input;
}

Tensor mkldnn_relu_backward(const Tensor& grad_output, const Tensor& input, const Scalar& threshold) {
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor grady = itensor_from_mkldnn(grad_output);
  ideep::tensor gradx;
  ideep::eltwise_backward::compute(x, grady, gradx,
      ideep::algorithm::eltwise_relu, /*alpha*/ 0.0);
  return new_with_itensor_mkldnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

}

#endif // AT_MKLDNN_ENABLED

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Config.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/relu_native.h`
- `ATen/ops/threshold_backward_native.h`
- `ATen/native/mkldnn/MKLDNNCommon.h`
- `ATen/native/mkldnn/Utils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`aten/src/ATen/native/mkldnn`):

- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`Gelu.cpp_docs.md`](./Gelu.cpp_docs.md)
- [`Conv.h_docs.md`](./Conv.h_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`Matmul.cpp_docs.md`](./Matmul.cpp_docs.md)
- [`TensorShape.cpp_docs.md`](./TensorShape.cpp_docs.md)
- [`RNN.cpp_docs.md`](./RNN.cpp_docs.md)
- [`RegisterMkldnnOpContextClass.cpp_docs.md`](./RegisterMkldnnOpContextClass.cpp_docs.md)
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)


## Cross-References

- **File Documentation**: `Relu.cpp_docs.md`
- **Keyword Index**: `Relu.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
