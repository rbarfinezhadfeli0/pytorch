# Documentation: `docs/aten/src/ATen/native/metal/MetalPrepackOpContext.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/metal/MetalPrepackOpContext.h_docs.md`
- **Size**: 7,245 bytes (7.08 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/metal/MetalPrepackOpContext.h`

## File Metadata

- **Path**: `aten/src/ATen/native/metal/MetalPrepackOpContext.h`
- **Size**: 4,637 bytes (4.53 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Tensor.h>
#include <torch/custom_class.h>

namespace at::native::metal {

using SerializationTypeConv2dPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    std::optional<Scalar>,
    std::optional<Scalar>>;

class Conv2dOpContext : public torch::jit::CustomClassHolder {
 public:
  SerializationTypeConv2dPrePack pack() {
    return std::make_tuple(
        weight_,
        bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        output_min_,
        output_max_);
  }
  Conv2dOpContext() = delete;
  Conv2dOpContext(
      at::Tensor&& weight,
      std::optional<at::Tensor>&& bias,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      int64_t groups,
      std::optional<Scalar> output_min,
      std::optional<Scalar> output_max)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        output_min_(std::move(output_min)),
        output_max_(std::move(output_max)) {}

  ~Conv2dOpContext() override {
    if (releaseCallback_) {
      releaseCallback_(conv2dOp_);
    }
  }

  void release_resources() override {
    if (releaseCallback_) {
      releaseCallback_(conv2dOp_);
    }
  }

  const Tensor& get_weight() const {
    return weight_;
  }

  const std::optional<Tensor>& get_bias() const {
    return bias_;
  }

  const std::vector<int64_t>& get_stride() const {
    return stride_;
  }

  const std::vector<int64_t>& get_padding() const {
    return padding_;
  }

  const std::vector<int64_t>& get_dilation() const {
    return dilation_;
  }

  int64_t get_groups() const {
    return groups_;
  }

  const std::optional<Scalar>& get_output_min() const {
    return output_min_;
  }

  const std::optional<Scalar>& get_output_max() const {
    return output_max_;
  }

  void set_conv2dOpPtr(void* ptr) {
      conv2dOp_ = ptr;
  }

  void* get_conv2dOpPtr() const {
    return conv2dOp_;
  }

  void set_releaseCallback(const std::function<void(void*)>& func) {
    releaseCallback_ = func;
  }

  std::function<void(void*)>& get_releaseCallback() {
     return releaseCallback_;
  }

  private:
    Tensor weight_;
    std::optional<Tensor> bias_;
    std::vector<int64_t> stride_;
    std::vector<int64_t> padding_;
    std::vector<int64_t> dilation_;
    int64_t groups_;
    std::optional<Scalar> output_min_;
    std::optional<Scalar> output_max_;
    std::function<void(void*)> releaseCallback_ = nullptr;
    void* conv2dOp_ = nullptr; // reserved to hold MPSCNNConv2dOp objects
};

using SerializationTypeLinearPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::optional<Scalar>,
    std::optional<Scalar>>;

class LinearOpContext : public torch::jit::CustomClassHolder {
 public:
  SerializationTypeLinearPrePack pack() {
    return std::make_tuple(weight_, bias_, output_min_, output_max_);
  }
  LinearOpContext() = delete;
  LinearOpContext(
      at::Tensor&& weight,
      std::optional<at::Tensor>&& bias,
      std::optional<Scalar> output_min,
      std::optional<Scalar> output_max)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        output_min_(std::move(output_min)),
        output_max_(std::move(output_max)) {}

  ~LinearOpContext() override {
    if (releaseCallback_) {
      releaseCallback_(opaqueOpPtr_);
    }
  }

  void release_resources() override {
    if (releaseCallback_) {
      releaseCallback_(opaqueOpPtr_);
    }
  }

  const Tensor& get_weight() const {
    return weight_;
  }

  const std::optional<Tensor>& get_bias() const {
    return bias_;
  }

  const std::optional<Scalar>& get_output_min() const {
    return output_min_;
  }

  const std::optional<Scalar>& get_output_max() const {
    return output_max_;
  }

  void set_opaqueOpPtr(void* ptr) {
    opaqueOpPtr_ = ptr;
  }

  void* get_opaqueOpPtr() const {
    return opaqueOpPtr_;
  }

  void set_releaseCallback(const std::function<void(void*)>& func) {
    releaseCallback_ = func;
  }

  std::function<void(void*)>& get_releaseCallback() {
    return releaseCallback_;
  }

 private:
  Tensor weight_;
  std::optional<Tensor> bias_;
  std::optional<Scalar> output_min_;
  std::optional<Scalar> output_max_;
  void* opaqueOpPtr_ = nullptr; // reserved to hold MPSCNNFullyConnected objects
  std::function<void(void*)> releaseCallback_ = nullptr;
};

} // namespace at::native::metal

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Conv2dOpContext`, `LinearOpContext`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/metal`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `torch/custom_class.h`


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

Files in the same folder (`aten/src/ATen/native/metal`):

- [`MetalShaders.h_docs.md`](./MetalShaders.h_docs.md)
- [`MetalDevice.h_docs.md`](./MetalDevice.h_docs.md)
- [`MetalGuardImpl.cpp_docs.md`](./MetalGuardImpl.cpp_docs.md)
- [`MetalNeuronType.h_docs.md`](./MetalNeuronType.h_docs.md)
- [`MetalTensorImplStorage.h_docs.md`](./MetalTensorImplStorage.h_docs.md)
- [`MetalConvParams.h_docs.md`](./MetalConvParams.h_docs.md)
- [`MetalPrepackOpRegister.cpp_docs.md`](./MetalPrepackOpRegister.cpp_docs.md)
- [`MetalCommandBuffer.h_docs.md`](./MetalCommandBuffer.h_docs.md)
- [`MetalTensorUtils.h_docs.md`](./MetalTensorUtils.h_docs.md)
- [`MetalTensorImpl.h_docs.md`](./MetalTensorImpl.h_docs.md)


## Cross-References

- **File Documentation**: `MetalPrepackOpContext.h_docs.md`
- **Keyword Index**: `MetalPrepackOpContext.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/metal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/metal`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/metal`):

- [`MetalShaders.h_docs.md_docs.md`](./MetalShaders.h_docs.md_docs.md)
- [`MetalCommandBuffer.h_kw.md_docs.md`](./MetalCommandBuffer.h_kw.md_docs.md)
- [`MetalTensorImplStorage.h_kw.md_docs.md`](./MetalTensorImplStorage.h_kw.md_docs.md)
- [`MetalDevice.h_docs.md_docs.md`](./MetalDevice.h_docs.md_docs.md)
- [`MetalNeuronType.h_kw.md_docs.md`](./MetalNeuronType.h_kw.md_docs.md)
- [`MetalCommandBuffer.h_docs.md_docs.md`](./MetalCommandBuffer.h_docs.md_docs.md)
- [`MetalTensorImplStorage.h_docs.md_docs.md`](./MetalTensorImplStorage.h_docs.md_docs.md)
- [`MetalPrepackOpRegister.cpp_docs.md_docs.md`](./MetalPrepackOpRegister.cpp_docs.md_docs.md)
- [`MetalNeuronType.h_docs.md_docs.md`](./MetalNeuronType.h_docs.md_docs.md)
- [`MetalShaders.h_kw.md_docs.md`](./MetalShaders.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `MetalPrepackOpContext.h_docs.md_docs.md`
- **Keyword Index**: `MetalPrepackOpContext.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
