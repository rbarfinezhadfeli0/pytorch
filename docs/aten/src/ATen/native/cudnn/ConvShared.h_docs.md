# Documentation: `aten/src/ATen/native/cudnn/ConvShared.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cudnn/ConvShared.h`
- **Size**: 5,032 bytes (4.91 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/Tensor.h>

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/native/ConvUtils.h>

namespace at {
namespace native {

// ---------------------------------------------------------------------
//
// Helper classes
//
// ---------------------------------------------------------------------

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams {
  c10::DeviceIndex device_id;
  cudnnDataType_t dataType;
  int input_size[2 + max_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  bool allow_tf32;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

std::ostream& operator<<(std::ostream& out, const ConvolutionParams& params);

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool deterministic,
    bool allow_tf32,
    at::MemoryFormat memory_format);

std::string repro_from_args(const ConvolutionParams& args);

// ---------------------------------------------------------------------
//
// Raw functions
//
// ---------------------------------------------------------------------

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
    bool allow_tf32);

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
    bool allow_tf32);

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
    bool allow_tf32);

void raw_cudnn_convolution_add_relu_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_add_relu_fallback_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

#if AT_CUDNN_ENABLED()

// v7 functions are preserved here to allow for runtime switching to v7
// (e.g., TORCH_CUDNN_V8_API_DISABLED=1).
// Note that v7 forward/backward out can have different behavior from the v8
// versions, as v7 explicitly splits large tensors as a 32-bit indexing
// workaround whereas v8 expects cuDNN to handle large tensors.
void raw_cudnn_convolution_forward_out_v7(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_backward_input_out_v7(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_backward_weight_out_v7(
    const Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

void raw_cudnn_convolution_add_relu_out_v7(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);
#endif
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `native`, `at`

**Classes/Structs**: `is`, `ConvolutionParams`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/cudnn/Descriptors.h`
- `ATen/cudnn/Types.h`
- `ATen/cudnn/cudnn-wrapper.h`
- `ATen/native/ConvUtils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
- [`RNNUtils.h_docs.md`](./RNNUtils.h_docs.md)
- [`BatchNorm.h_docs.md`](./BatchNorm.h_docs.md)
- [`Conv_v8.cpp_docs.md`](./Conv_v8.cpp_docs.md)
- [`GridSampler.cpp_docs.md`](./GridSampler.cpp_docs.md)


## Cross-References

- **File Documentation**: `ConvShared.h_docs.md`
- **Keyword Index**: `ConvShared.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
