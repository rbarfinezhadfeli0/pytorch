# Documentation: `torch/csrc/jit/tensorexpr/operators/quantization.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/operators/quantization.h`
- **Size**: 5,289 bytes (5.17 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch::jit::tensorexpr {

TORCH_API ExprHandle quantizePerTensorQParamFromArg(ArgValue arg);

TORCH_API double immQScale(const BufHandle& qx);

TORCH_API int64_t immQZero(const BufHandle& qx);

TORCH_API ScalarType immQDType(const BufHandle& qx);

TORCH_API bool isQuantized(const BufHandle& qx);

TORCH_API Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizePerTensorExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedConv1d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedConv2dPrepack(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedConv2dRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedLinear(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedLinearRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedAdd(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedAddExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedMul(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedMulScalar(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedCat(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeDequantizeExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeUpsampleNearest2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeUpsampleNearest2dExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedSigmoidExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device /*unused*/);
} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 24 function(s).

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

- `torch/csrc/jit/tensorexpr/kernel.h`


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
- [`conv2d.h_docs.md`](./conv2d.h_docs.md)
- [`misc.cpp_docs.md`](./misc.cpp_docs.md)
- [`softmax.cpp_docs.md`](./softmax.cpp_docs.md)
- [`misc.h_docs.md`](./misc.h_docs.md)
- [`quantization.cpp_docs.md`](./quantization.cpp_docs.md)
- [`conv2d.cpp_docs.md`](./conv2d.cpp_docs.md)
- [`softmax.h_docs.md`](./softmax.h_docs.md)
- [`pointwise.cpp_docs.md`](./pointwise.cpp_docs.md)
- [`matmul.h_docs.md`](./matmul.h_docs.md)


## Cross-References

- **File Documentation**: `quantization.h_docs.md`
- **Keyword Index**: `quantization.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
