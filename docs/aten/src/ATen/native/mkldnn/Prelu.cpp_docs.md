# Documentation: `aten/src/ATen/native/mkldnn/Prelu.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/Prelu.cpp`
- **Size**: 2,714 bytes (2.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>


#if !AT_MKLDNN_ENABLED()

namespace at::native {

Tensor mkldnn_prelu(const Tensor& input, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_prelu: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_prelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_prelu_backward: ATen not compiled with MKLDNN support");
}

}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at::native {

Tensor mkldnn_prelu(const Tensor& input, const Tensor& weight) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  const ideep::tensor& x = itensor_from_mkldnn(input);
  const ideep::tensor& w = itensor_from_tensor(weight);

  ideep::tensor y;
  ideep::prelu_forward::compute(
      x, w, y, ideep::prop_kind::forward_training);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

std::tuple<Tensor, Tensor> mkldnn_prelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) {
  const ideep::tensor& x = itensor_from_mkldnn(input);
  const ideep::tensor& w = itensor_from_tensor(weight);
  const ideep::tensor grady = itensor_from_mkldnn(grad_output);
  ideep::tensor gradx;
  ideep::tensor gradw;

  ideep::prelu_backward::compute(
      x, w, grady, gradx, gradw, ideep::prop_kind::backward);
  if (weight.is_mkldnn()) {
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(gradx),
                                optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                grad_output.options().device_opt()),
        new_with_itensor_mkldnn(std::move(gradw),
                                optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()));
  } else {
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(gradx),
                                optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                grad_output.options().device_opt()),
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw),
                                                optTypeMetaToScalarType(weight.options().dtype_opt()),
                                                weight.options().device_opt())));
  }
}
}

#endif // AT_MKLDNN_ENABLED

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `ATen/ATen.h`
- `ATen/NativeFunctions.h`
- `ATen/Config.h`
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

- **File Documentation**: `Prelu.cpp_docs.md`
- **Keyword Index**: `Prelu.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
