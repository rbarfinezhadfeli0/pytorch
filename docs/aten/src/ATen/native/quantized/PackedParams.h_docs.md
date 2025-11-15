# Documentation: `aten/src/ATen/native/quantized/PackedParams.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/PackedParams.h`
- **Size**: 4,739 bytes (4.63 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>

struct LinearPackedParamsBase : public torch::jit::CustomClassHolder {
  virtual at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;

  // out variant of LinearPackedParamsBase::apply
  virtual at::Tensor& apply_out(
      const at::Tensor& /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/,
      at::Tensor& output) {
    throw std::runtime_error(
        "apply_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  virtual at::Tensor& apply_relu_out(
      const at::Tensor& /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/,
      at::Tensor& output) {
    throw std::runtime_error(
        "apply_relu_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  // Corresponding pattern (the ops with `*` are part of the pattern that
  // represents the computation of quantized::linear_with_input_q_dq_qweight_dq_output_fp32):
  // input -> q* -> dq* -> linear* ->
  //         qweight -> dq* /
  //
  // After fusion:
  // input -> quantized::linear_with_input_q_dq_qweight_dq_output_fp32* ->
  //         qweight /
  //
  // Additional Note: the weight is packed as well
  // Params:
  //    X: float32 Tensor, will be quantized to quint8 in the op
  //    W_prepack: packed qint8 quantized weight and bias
  // Returns:
  //    Y: float32 Tensor
  virtual at::Tensor apply_with_input_q_dq_qweight_dq_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) {
    throw std::runtime_error(
        "apply_with_input_q_dq_qweight_dq_output_fp32 is not implemented for this packed "
        "parameter type");
    return {};
  }

  // Corresponding pattern (the ops with `*` are part of the pattern that
  // represents the computation of quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32):
  // input -> q* -> dq* -> linear* -> relu* ->
  //         qweight -> dq* /
  //
  // After fusion:
  // input -> quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32* ->
  //         qweight /
  //
  // Additional Note: the weight is packed as well
  // Params:
  //    input: float32 Tensor, will be quantized to quint8 in the op
  // Returns:
  //    float32 Tensor
  virtual at::Tensor apply_with_input_q_dq_qweight_dq_relu_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) {
    throw std::runtime_error(
        "apply_with_input_q_dq_qweight_dq_relu_output_fp32 is not implemented for this packed "
        "parameter type");
    return {};
  }

  virtual at::Tensor apply_dynamic(
      at::Tensor input,
      bool reduce_range = false) = 0;
  virtual at::Tensor apply_dynamic_relu(
      at::Tensor input,
      bool reduce_range = false) = 0;

  virtual at::Tensor& apply_dynamic_out(
      const at::Tensor& /* input */,
      at::Tensor& output,
      bool /* reduce_range */) {
    throw std::runtime_error(
        "apply_dynamic_out is not implemented for this packed "
        "parameter type");
    return output;
  }
  virtual at::Tensor& apply_dynamic_relu_out(
      const at::Tensor& /* input */,
      at::Tensor& output,
      bool /* reduce_range */) {
    throw std::runtime_error(
        "apply_dynamic_relu_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  virtual std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() = 0;

  virtual std::optional<at::Tensor> bias() = 0;

  virtual void set_bias(std::optional<at::Tensor> /*bias*/) {
    throw std::runtime_error(
        "set_bias is not implemented for this packed "
        "parameter type");
  }
};

template <int kSpatialDim = 2>
struct ConvPackedParamsBase : public torch::jit::CustomClassHolder {
  virtual at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_dynamic(
      const at::Tensor& input,
      bool reduce_range) = 0;

  virtual std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() = 0;

  virtual torch::List<int64_t> stride() const = 0;
  virtual torch::List<int64_t> padding() const = 0;
  virtual torch::List<int64_t> output_padding() const = 0;
  virtual torch::List<int64_t> dilation() const = 0;
  virtual int64_t groups() const = 0;
  virtual bool transpose() const = 0;
};

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `LinearPackedParamsBase`, `ConvPackedParamsBase`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/core/ivalue.h`


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

Files in the same folder (`aten/src/ATen/native/quantized`):

- [`QTensor.cpp_docs.md`](./QTensor.cpp_docs.md)
- [`FakeQuantAffine.h_docs.md`](./FakeQuantAffine.h_docs.md)
- [`qconv_unpack.cpp_docs.md`](./qconv_unpack.cpp_docs.md)
- [`AffineQuantizerBase.h_docs.md`](./AffineQuantizerBase.h_docs.md)
- [`AffineQuantizerBase.cpp_docs.md`](./AffineQuantizerBase.cpp_docs.md)
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)
- [`AffineQuantizer.h_docs.md`](./AffineQuantizer.h_docs.md)
- [`qlinear_unpack.cpp_docs.md`](./qlinear_unpack.cpp_docs.md)
- [`TensorFactories.cpp_docs.md`](./TensorFactories.cpp_docs.md)


## Cross-References

- **File Documentation**: `PackedParams.h_docs.md`
- **Keyword Index**: `PackedParams.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
