# Documentation: `docs/aten/src/ATen/native/mkldnn/RegisterMkldnnOpContextClass.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/RegisterMkldnnOpContextClass.cpp_docs.md`
- **Size**: 7,324 bytes (7.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/RegisterMkldnnOpContextClass.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/RegisterMkldnnOpContextClass.cpp`
- **Size**: 4,862 bytes (4.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

#include <ATen/Tensor.h>
#include <ATen/native/mkldnn/ConvPrepack.h>
#include <ATen/native/mkldnn/OpContext.h>
#include <ATen/native/mkldnn/Utils.h>
#include <torch/custom_class.h>
#include <torch/library.h>

namespace at::native::mkldnn {

using namespace internal::convolution;

static bool is_mkldnn_bf16_supported() {
#if defined(__aarch64__)
  return mkldnn_bf16_device_check_arm();
#else
  return mkldnn_bf16_device_check();
#endif
}

static bool is_mkldnn_fp16_supported() {
  return mkldnn_fp16_device_check();
}

static constexpr bool is_mkldnn_acl_supported() {
  return AT_MKLDNN_ACL_ENABLED();
}

TORCH_LIBRARY(mkldnn, m) {
  m.class_<ConvOpContext>(TORCH_SELECTIVE_CLASS("ConvOpContext"))
      .def_pickle(
          [](const c10::intrusive_ptr<ConvOpContext>& op_context)
              -> SerializationTypeConvPrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeConvPrePack state)
              -> c10::intrusive_ptr<ConvOpContext> { // __setstate__
            return std::apply(createConvPrePackOpContext, std::move(state));
          });

  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_linear_pointwise(Tensor X, Tensor W, Tensor? B, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_linear_pointwise.binary(Tensor X, Tensor other, Tensor W, Tensor? B, str attr) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_convolution_pointwise(Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_convolution_pointwise.binary(Tensor X, Tensor other, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_convolution_pointwise_.binary(Tensor(a!) other, Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor(a!) Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_convolution_transpose_pointwise(Tensor X, Tensor W, Tensor? B, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_reorder_convolution_transpose_weight(Tensor self, int[2] padding=0, int[2] output_padding=0, int[2] stride=1, int[2] dilation=1, int groups=1, int[]? input_size=None) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_reorder_linear_weight(Tensor self, int? batch_size=None) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_reorder_convolution_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1, int[]? input_size=None) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_reorder_mkldnn_rnn_layer_weight(Tensor weight0, Tensor weight1, int hidden_size, bool reverse, bool has_biases, bool batch_first, int[]? input_size=None) -> Tensor[] Y"));
  m.def("_is_mkldnn_bf16_supported", &is_mkldnn_bf16_supported);
  m.def("_is_mkldnn_fp16_supported", &is_mkldnn_fp16_supported);
  m.def("_is_mkldnn_acl_supported", &is_mkldnn_acl_supported);
  m.def("mkldnn::data_ptr(Tensor mkldnn_tensor) -> int");
  m.def("mkldnn::_get_mkldnn_serialized_md (Tensor mkldnn_tensor) -> Tensor");
  m.def("mkldnn::_nbytes(Tensor mkldnn_tensor) -> int");
}

TORCH_LIBRARY(mkldnn_prepacked, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, int[4] input_size, str attr) -> __torch__.torch.classes.mkldnn.ConvOpContext"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_run(Tensor X, __torch__.torch.classes.mkldnn.ConvOpContext W_prepack) -> Tensor Y"));
}

TORCH_LIBRARY_IMPL(mkldnn_prepacked, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_prepack"),
      TORCH_FN(createConvPrePackOpContext));

  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_run"), TORCH_FN(conv_run));
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()

#if AT_MKL_ENABLED() && AT_MKLDNN_ENABLED()

namespace at::native::mkl {

TORCH_LIBRARY(mkl, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkl::_mkl_reorder_linear_weight(Tensor X, int batch_size) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkl::_mkl_linear(Tensor X, Tensor MKL_W, Tensor ORI_W, Tensor? B, int batch_size) -> Tensor"));
}

} // namespace at::native::mkl

#endif // AT_MKL_ENABLED && AT_MKLDNN_ENABLED

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `internal`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/Tensor.h`
- `ATen/native/mkldnn/ConvPrepack.h`
- `ATen/native/mkldnn/OpContext.h`
- `ATen/native/mkldnn/Utils.h`
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

- **Serialization**: Uses pickle - be cautious with untrusted data

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
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)


## Cross-References

- **File Documentation**: `RegisterMkldnnOpContextClass.cpp_docs.md`
- **Keyword Index**: `RegisterMkldnnOpContextClass.cpp_kw.md`
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

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

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
- [`BinaryOps.cpp_docs.md_docs.md`](./BinaryOps.cpp_docs.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`MkldnnTensorMath.cpp_kw.md_docs.md`](./MkldnnTensorMath.cpp_kw.md_docs.md)
- [`SoftMax.cpp_docs.md_docs.md`](./SoftMax.cpp_docs.md_docs.md)
- [`ConvPrepack.cpp_kw.md_docs.md`](./ConvPrepack.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `RegisterMkldnnOpContextClass.cpp_docs.md_docs.md`
- **Keyword Index**: `RegisterMkldnnOpContextClass.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
