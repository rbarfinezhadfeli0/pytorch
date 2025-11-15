# Documentation: `aten/src/ATen/native/cudnn/ConvPlaceholders.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cudnn/ConvPlaceholders.cpp`
- **Size**: 6,488 bytes (6.34 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h> // for the definition of AT_CUDNN_ENABLED

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cudnn_convolution_add_relu_native.h>
#include <ATen/ops/cudnn_convolution_native.h>
#include <ATen/ops/cudnn_convolution_relu_native.h>
#include <ATen/ops/cudnn_convolution_transpose_native.h>
#endif

namespace at::native {

// ---------------------------------------------------------------------
//
// Placeholder operators
//
// ---------------------------------------------------------------------

#if !AT_CUDNN_ENABLED()

// See Note [ATen preprocessor philosophy]

at::Tensor cudnn_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(false, "cudnn_convolution: ATen not compiled with cuDNN support");
}

at::Tensor& cudnn_convolution_out(
    const Tensor& input_t,
    const Tensor& weight_t,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    Tensor& output_t) {
  TORCH_CHECK(
      false, "cudnn_convolution_out: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_backward_input(
    IntArrayRef input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "cudnn_convolution_backward_input: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_backward_weight(
    IntArrayRef weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "cudnn_convolution_backward_weight: ATen not compiled with cuDNN support");
}

std::tuple<at::Tensor, at::Tensor> cudnn_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    std::array<bool, 2> output_mask) {
  TORCH_CHECK(
      false,
      "cudnn_convolution_backward: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose(
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "cudnn_convolution_transpose: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose_backward_input(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose_backward_weight(
    IntArrayRef weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "cudnn_convolution_transpose_backward_weight: ATen not compiled with cuDNN support");
}

std::tuple<at::Tensor, at::Tensor> cudnn_convolution_transpose_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    std::array<bool, 2> output_mask) {
  TORCH_CHECK(
      false,
      "cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
}

void raw_cudnn_convolution_forward_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "raw_cudnn_convolution_forward_out: ATen not compiled with cuDNN support");
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "raw_cudnn_convolution_backward_input_out: ATen not compiled with cuDNN support");
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "raw_cudnn_convolution_backward_weight_out: ATen not compiled with cuDNN support");
}

Tensor cudnn_convolution_relu(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_t,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(
      false, "cudnn_convolution_relu: ATen not compiled with cuDNN support");
}

Tensor cudnn_convolution_add_relu(
    const Tensor& input_t,
    const Tensor& weight_t,
    const Tensor& z_t,
    const std::optional<Scalar>& alpha,
    const std::optional<Tensor>& bias_t,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(
      false,
      "cudnn_convolution_add_relu: ATen not compiled with cuDNN support");
}

#endif // AT_CUDNN_ENABLED

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

This file is located in `aten/src/ATen/native/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/cuda/CUDAConfig.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/cudnn_convolution_add_relu_native.h`
- `ATen/ops/cudnn_convolution_native.h`
- `ATen/ops/cudnn_convolution_relu_native.h`
- `ATen/ops/cudnn_convolution_transpose_native.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/native/cudnn`):

- [`MHA.cpp_docs.md`](./MHA.cpp_docs.md)
- [`Conv_v7.cpp_docs.md`](./Conv_v7.cpp_docs.md)
- [`RNN.cpp_docs.md`](./RNN.cpp_docs.md)
- [`ConvShared.cpp_docs.md`](./ConvShared.cpp_docs.md)
- [`MHA.h_docs.md`](./MHA.h_docs.md)
- [`ConvShared.h_docs.md`](./ConvShared.h_docs.md)
- [`RNNUtils.h_docs.md`](./RNNUtils.h_docs.md)
- [`BatchNorm.h_docs.md`](./BatchNorm.h_docs.md)
- [`Conv_v8.cpp_docs.md`](./Conv_v8.cpp_docs.md)
- [`GridSampler.cpp_docs.md`](./GridSampler.cpp_docs.md)


## Cross-References

- **File Documentation**: `ConvPlaceholders.cpp_docs.md`
- **Keyword Index**: `ConvPlaceholders.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
