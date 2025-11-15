# Documentation: KernelRegistry.h

## File Metadata
- **Path**: `torch/nativert/kernels/KernelRegistry.h`
- **Size**: 5070 bytes
- **Lines**: 120
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
#pragma once

#include <ATen/ATen.h>

#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/kernels/PrimKernelRegistry.h>

namespace torch::nativert {

TORCH_DECLARE_REGISTRY(
    StaticallyDispatchedCPUKernelRegistry,
    OpKernel,
    const Node*);

#define REGISTER_CPU_KERNEL(name, id, ...)                                \
  class OpKernel_##id : public C10Kernel {                                \
   public:                                                                \
    OpKernel_##id(const Node* node)                                       \
        : C10Kernel(                                                      \
              node,                                                       \
              torch::nativert::OpKernelKind::kStaticDispatchKernel) {}    \
    void computeInternal(torch::nativert::ExecutionFrame& executionFrame) \
        const override final {                                            \
      __VA_ARGS__;                                                        \
    }                                                                     \
  };                                                                      \
  C10_REGISTER_TYPED_CLASS(                                               \
      StaticallyDispatchedCPUKernelRegistry, name, OpKernel_##id)

#define ALIASING_SPEC(...) __VA_ARGS__

#define REGISTER_ALIASING_CPU_KERNEL(name, id, aliasing_spec, ...)        \
  class OpKernel_##id : public C10Kernel {                                \
   public:                                                                \
    OpKernel_##id(const Node* node)                                       \
        : C10Kernel(                                                      \
              node,                                                       \
              torch::nativert::OpKernelKind::kNativeStaticDispatchKernel, \
              aliasing_spec) {}                                           \
    void computeInternal(torch::nativert::ExecutionFrame& executionFrame) \
        const override final {                                            \
      __VA_ARGS__;                                                        \
    }                                                                     \
  };                                                                      \
  C10_REGISTER_TYPED_CLASS(                                               \
      StaticallyDispatchedCPUKernelRegistry, name, OpKernel_##id)

#define REGISTER_NATIVE_CPU_KERNEL(name, id, ...)                            \
  class OpKernel_##id : public C10Kernel {                                   \
   public:                                                                   \
    OpKernel_##id(const Node* node)                                          \
        : C10Kernel(                                                         \
              node,                                                          \
              torch::nativert::OpKernelKind::kNativeStaticDispatchKernel) {} \
    void computeInternal(torch::nativert::ExecutionFrame& executionFrame)    \
        const override final {                                               \
      __VA_ARGS__;                                                           \
    }                                                                        \
  };                                                                         \
  C10_REGISTER_TYPED_CLASS(                                                  \
      StaticallyDispatchedCPUKernelRegistry, name, OpKernel_##id)

inline at::Tensor create_empty_from(const at::Tensor& t) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      std::nullopt,
      std::nullopt);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype) {
  return at::detail::empty_cpu(
      {0}, dtype, t.layout(), t.device(), std::nullopt, std::nullopt);
}

inline at::Tensor create_empty_from(const at::Tensor& t, c10::Device device) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      device,
      std::nullopt,
      std::nullopt);
}
inline at::Tensor create_empty_from(const at::Tensor& t, c10::Layout layout) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      layout,
      t.device(),
      std::nullopt,
      std::nullopt);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::MemoryFormat memory_format) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      std::nullopt,
      memory_format);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype,
    c10::MemoryFormat memory_format) {
  return at::detail::empty_cpu(
      {0}, dtype, t.layout(), t.device(), std::nullopt, memory_format);
}

} // namespace torch::nativert

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 3 class(es): OpKernel_, OpKernel_, OpKernel_


## Key Components

The file contains 271 words across 120 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5070 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
