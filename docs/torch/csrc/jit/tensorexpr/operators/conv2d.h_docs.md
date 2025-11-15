# Documentation: `torch/csrc/jit/tensorexpr/operators/conv2d.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/operators/conv2d.h`
- **Size**: 2,893 bytes (2.83 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch::jit::tensorexpr {

// An API to compute 2D depthwise convolutions with bias.
TORCH_API Tensor conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    int stride,
    int pad,
    int groups);

// An API to compute 2D depthwise convolutions without bias.
TORCH_API Tensor conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    int stride,
    int pad,
    int groups);

TORCH_API Tensor conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups);

TORCH_API Tensor conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups);

bool conv2dIsSupported(
    const TensorInfo& input,
    const TensorInfo& weight,
    const TensorInfo& bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& pad,
    const std::vector<int64_t>& dilation,
    int64_t groups);
bool mkldnnPrepackedConvIsSupported(
    const TensorInfo& input,
    const TensorInfo& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& pad,
    const std::vector<int64_t>& dilation,
    int64_t groups);
Tensor computeConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeConv1d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);
Tensor computePrepackedConv2dClampRun(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);
Tensor computePrepackedLinearClampRun(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeMkldnnPrepackedConvRun(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);
} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr/operators`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/tensorexpr/operators/misc.h`
- `torch/csrc/jit/tensorexpr/tensor.h`


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

Files in the same folder (`torch/csrc/jit/tensorexpr/operators`):

- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`misc.cpp_docs.md`](./misc.cpp_docs.md)
- [`softmax.cpp_docs.md`](./softmax.cpp_docs.md)
- [`misc.h_docs.md`](./misc.h_docs.md)
- [`quantization.cpp_docs.md`](./quantization.cpp_docs.md)
- [`conv2d.cpp_docs.md`](./conv2d.cpp_docs.md)
- [`softmax.h_docs.md`](./softmax.h_docs.md)
- [`pointwise.cpp_docs.md`](./pointwise.cpp_docs.md)
- [`matmul.h_docs.md`](./matmul.h_docs.md)


## Cross-References

- **File Documentation**: `conv2d.h_docs.md`
- **Keyword Index**: `conv2d.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
