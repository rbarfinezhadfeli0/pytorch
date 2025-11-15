# Documentation: `docs/aten/src/ATen/native/mkldnn/OpContext.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/OpContext.h_docs.md`
- **Size**: 4,825 bytes (4.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/OpContext.h`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/OpContext.h`
- **Size**: 2,359 bytes (2.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/ivalue.h>
#include <ATen/native/mkldnn/Common.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::mkldnn {

const static std::map<std::string, ideep::attr_t> fusion_attr_map = {
    {"none", ideep::attr_t()},
    {"relu", ideep::attr_t::fuse_relu()},
};

using SerializationTypeConvPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    std::vector<int64_t>,
    std::string>;

class ConvOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  std::optional<Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  std::vector<int64_t> input_size_;
  std::string attr_;

 public:
  SerializationTypeConvPrePack unpack() {
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        input_size_,
        attr_);
  }

  virtual Tensor run(const Tensor& input) = 0;
  virtual void run(const Tensor& input, void* output) = 0;
};

class MkldnnConvOpContext final : public ConvOpContext {
 private:
  ContextConv op_context_;

 public:
  MkldnnConvOpContext(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      std::vector<int64_t>&& input_size,
      ContextConv&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    padding_ = std::move(padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    groups_ = groups;
    input_size_ = std::move(input_size);
  }

  Tensor run(const Tensor& input) override;

  void run(const Tensor& input, void* output) override;

  static c10::intrusive_ptr<ConvOpContext> create_context(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      std::vector<int64_t>&& input_size,
      const ideep::attr_t& attr);
};

} // namespace at

#endif // AT_MKLDNN_ENABLED()

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `ConvOpContext`, `MkldnnConvOpContext`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `ATen/core/ivalue.h`
- `ATen/native/mkldnn/Common.h`


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

- **File Documentation**: `OpContext.h_docs.md`
- **Keyword Index**: `OpContext.h_kw.md`
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
- [`BinaryOps.cpp_docs.md_docs.md`](./BinaryOps.cpp_docs.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`MkldnnTensorMath.cpp_kw.md_docs.md`](./MkldnnTensorMath.cpp_kw.md_docs.md)
- [`SoftMax.cpp_docs.md_docs.md`](./SoftMax.cpp_docs.md_docs.md)
- [`ConvPrepack.cpp_kw.md_docs.md`](./ConvPrepack.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `OpContext.h_docs.md_docs.md`
- **Keyword Index**: `OpContext.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
