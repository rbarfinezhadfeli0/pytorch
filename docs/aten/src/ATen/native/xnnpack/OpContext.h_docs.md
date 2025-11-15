# Documentation: `aten/src/ATen/native/xnnpack/OpContext.h`

## File Metadata

- **Path**: `aten/src/ATen/native/xnnpack/OpContext.h`
- **Size**: 7,270 bytes (7.10 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#ifdef USE_XNNPACK

#include <ATen/core/ivalue.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/Tensor.h>

namespace at::native::xnnpack {

using SerializationTypeLinearPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::optional<Scalar>,
    std::optional<Scalar>>;
using SerializationTypeConv2dPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    std::optional<Scalar>,
    std::optional<Scalar>>;
using SerializationTypeTransposeConv2dPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    std::optional<Scalar>,
    std::optional<Scalar>>;



class LinearOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  std::optional<Tensor> orig_bias_;
  std::optional<Scalar> output_min_;
  std::optional<Scalar> output_max_;
  bool orig_weight_and_bias_freed_;

 public:
  SerializationTypeLinearPrePack unpack() {
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
    return std::make_tuple(orig_weight_, orig_bias_, output_min_, output_max_);
  }

  virtual Tensor run(const Tensor& input) = 0;
  virtual void free_orig_weight_and_bias() = 0;
};

class XNNPackLinearOpContext final : public LinearOpContext {
 private:
  ContextLinear op_context_;

 public:
  XNNPackLinearOpContext(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      const std::optional<Scalar>& min,
      const std::optional<Scalar>& max,
      ContextLinear&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    output_min_ = min;
    output_max_ = max;
    orig_weight_and_bias_freed_ = false;
  }

  Tensor run(const Tensor& input) override;
  void free_orig_weight_and_bias() override;

  static c10::intrusive_ptr<LinearOpContext> create_context(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      const std::optional<Scalar>& output_min,
      const std::optional<Scalar>& output_max);
};

class Conv2dOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  std::optional<Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  std::optional<Scalar> output_min_;
  std::optional<Scalar> output_max_;
  bool orig_weight_and_bias_freed_;

 public:
  SerializationTypeConv2dPrePack unpack() {
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        output_min_,
        output_max_);
  }

  virtual Tensor run(const Tensor& input) = 0;
  virtual void free_orig_weight_and_bias() = 0;
};

class TransposeConv2dOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  std::optional<Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> output_padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  std::optional<Scalar> output_min_;
  std::optional<Scalar> output_max_;
  bool orig_weight_and_bias_freed_;

 public:
  SerializationTypeTransposeConv2dPrePack unpack() {
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        output_padding_,
        dilation_,
        groups_,
        output_min_,
        output_max_);
  }

  virtual Tensor run(const Tensor& input) = 0;
  virtual void free_orig_weight_and_bias() = 0;
};

class XNNPackConv2dOpContext final : public Conv2dOpContext {
 private:
  ContextConv2D op_context_;
  // xnnpack convs use indirection buffer.
  // These buffers need setup at runtime and/or when input
  // dims change. If we are running the same model on multiple
  // threads, this can lead to contention where indirection buffer
  // is being accessed and updated at the same time from two different
  // threads.
  std::mutex xnnp_mutex_;

 public:
  XNNPackConv2dOpContext(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      const std::optional<Scalar>& min,
      const std::optional<Scalar>& max,
      ContextConv2D&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    padding_ = std::move(padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    groups_ = groups;
    output_min_ = min;
    output_max_ = max;
    orig_weight_and_bias_freed_ = false;
  }

  Tensor run(const Tensor& input) override;
  void free_orig_weight_and_bias() override;

  static c10::intrusive_ptr<Conv2dOpContext> create_context(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      const std::optional<Scalar>& output_min,
      const std::optional<Scalar>& output_max);
};

class XNNPackTransposeConv2dOpContext final : public TransposeConv2dOpContext {
 private:
  ContextConv2D op_context_;
  // xnnpack convs use indirection buffer.
  // These buffers need setup at runtime and/or when input
  // dims change. If we are running the same model on multiple
  // threads, this can lead to contention where indirection buffer
  // is being accessed and updated at the same time from two different
  // threads.
  std::mutex xnnp_mutex_;

 public:
  XNNPackTransposeConv2dOpContext(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& output_padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      const std::optional<Scalar>& min,
      const std::optional<Scalar>& max,
      ContextConv2D&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    padding_ = std::move(padding);
    output_padding_ = std::move(output_padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    groups_ = groups;
    output_min_ = min;
    output_max_ = max;
    orig_weight_and_bias_freed_ = false;
  }

  Tensor run(const Tensor& input) override;
  void free_orig_weight_and_bias() override;

  static c10::intrusive_ptr<TransposeConv2dOpContext> create_context(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& output_padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      const std::optional<Scalar>& output_min,
      const std::optional<Scalar>& output_max);
};

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

```



## High-Level Overview


This C++ file contains approximately 6 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `LinearOpContext`, `XNNPackLinearOpContext`, `Conv2dOpContext`, `TransposeConv2dOpContext`, `XNNPackConv2dOpContext`, `XNNPackTransposeConv2dOpContext`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `ATen/native/xnnpack/Common.h`
- `ATen/Tensor.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/native/xnnpack`):

- [`Engine.h_docs.md`](./Engine.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`ChannelShuffle.cpp_docs.md`](./ChannelShuffle.cpp_docs.md)
- [`Convolution.h_docs.md`](./Convolution.h_docs.md)
- [`RegisterOpContextClass.cpp_docs.md`](./RegisterOpContextClass.cpp_docs.md)
- [`Common.h_docs.md`](./Common.h_docs.md)
- [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`Linear.h_docs.md`](./Linear.h_docs.md)
- [`Shim.cpp_docs.md`](./Shim.cpp_docs.md)


## Cross-References

- **File Documentation**: `OpContext.h_docs.md`
- **Keyword Index**: `OpContext.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
