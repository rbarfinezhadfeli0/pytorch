# Documentation: `docs/aten/src/ATen/native/mkldnn/BinaryOps.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/BinaryOps.cpp_docs.md`
- **Size**: 7,450 bytes (7.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/BinaryOps.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/BinaryOps.cpp`
- **Size**: 4,900 bytes (4.79 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/mul_native.h>
#endif

#if !AT_MKLDNN_ENABLED()


namespace at::native {

Tensor& mkldnn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  TORCH_CHECK(false, "mkldnn_add_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "mkldnn_add: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "mkldnn_add_: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "mkldnn_mul_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul_: ATen not compiled with MKLDNN support");
}

} // namespace at::native


#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native {

static Tensor emptyBinaryOp(const Tensor& self, const Tensor& other) {
  if (!self.requires_grad() && !other.requires_grad()) {
    auto out_size = infer_size(self.sizes(), other.sizes());
    auto out_dtype = promoteTypes(
        c10::typeMetaToScalarType(self.dtype()),
        c10::typeMetaToScalarType(other.dtype()));
    TORCH_CHECK(
        self.device() == other.device(),
        "Expected same device for binary mkldnn op");
    return empty_mkldnn(
        out_size,
        out_dtype,
        self.options().layout_opt(),
        self.options().device_opt(),
        self.options().pinned_memory_opt());
  } else {
    TORCH_CHECK(
        false,
        "MKLDNN does not support Binary Ops with a 0-dimension Tensor in training");
  }
}

Tensor& mkldnn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor& z = itensor_from_mkldnn(result);
  if (result.is_same(other)) {
    const std::vector<float> scales{alpha.to<float>(), 1.0};
    ideep::sum::compute(scales, {y, x}, z);
  } else {
    const std::vector<float> scales{1.0, alpha.to<float>()};
    ideep::sum::compute(scales, {x, y}, z);
  }

  return result;
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }

  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor z;
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute(scales, {x, y}, z);

  return new_with_itensor_mkldnn(std::move(z), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return native::mkldnn_add_out(self, other, alpha, self);
}

Tensor& mkldnn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(result.sizes() == self.sizes(),
             "mkldnn_mul_out: the output size should be same as input size");
  ideep::tensor& z = itensor_from_mkldnn(result);
  ideep::tensor& x = itensor_from_mkldnn(self);

  // for zero_dim tensor
  if (other.ndimension() == 0) {
    ideep::eltwise_forward::compute(
      x, z, ideep::algorithm::eltwise_linear,
      ideep::prop_kind::forward_inference, /*alpha*/ other.item().to<float>());

    return result;
  } else {
    TORCH_CHECK(self.sizes() == other.sizes(),
               "mkldnn_mul_out: currently mkldnn not support broadcasting");
    ideep::tensor y = itensor_from_mkldnn(other);
    ideep::binary::compute(x, y, z, dnnl::algorithm::binary_mul);

    return result;
  }
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }
  Tensor result = empty_mkldnn(self.sizes(), optTypeMetaToScalarType(self.options().dtype_opt()),
                               self.options().layout_opt(), self.options().device_opt(),
                               self.options().pinned_memory_opt());
  return native::mkldnn_mul_out(self, other, result);
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  return native::mkldnn_mul_out(self, other, self);
}

} // namespace at

#endif // AT_MKLDNN_ENABLED

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

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
- `ATen/ExpandUtils.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/add_native.h`
- `ATen/ops/empty_native.h`
- `ATen/ops/mul_native.h`
- `ATen/native/mkldnn/MKLDNNCommon.h`


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

- **File Documentation**: `BinaryOps.cpp_docs.md`
- **Keyword Index**: `BinaryOps.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/mkldnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/mkldnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/aten/src/ATen/native/mkldnn`):

- [`ConvPrepack.h_docs.md_docs.md`](./ConvPrepack.h_docs.md_docs.md)
- [`Conv.h_kw.md_docs.md`](./Conv.h_kw.md_docs.md)
- [`IDeepRegistration.h_docs.md_docs.md`](./IDeepRegistration.h_docs.md_docs.md)
- [`Prelu.cpp_kw.md_docs.md`](./Prelu.cpp_kw.md_docs.md)
- [`MKLDNNConversions.cpp_kw.md_docs.md`](./MKLDNNConversions.cpp_kw.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`MkldnnTensorMath.cpp_kw.md_docs.md`](./MkldnnTensorMath.cpp_kw.md_docs.md)
- [`SoftMax.cpp_docs.md_docs.md`](./SoftMax.cpp_docs.md_docs.md)
- [`ConvPrepack.cpp_kw.md_docs.md`](./ConvPrepack.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `BinaryOps.cpp_docs.md_docs.md`
- **Keyword Index**: `BinaryOps.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
