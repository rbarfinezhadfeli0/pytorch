# Documentation: `docs/torch/nativert/kernels/KernelRegistry.h_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/kernels/KernelRegistry.h_docs.md`
- **Size**: 7,638 bytes (7.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/kernels/KernelRegistry.h`

## File Metadata

- **Path**: `torch/nativert/kernels/KernelRegistry.h`
- **Size**: 5,070 bytes (4.95 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
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


This C++ file contains approximately 3 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `OpKernel_`, `OpKernel_`, `OpKernel_`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `torch/nativert/executor/OpKernel.h`
- `torch/nativert/graph/Graph.h`
- `torch/nativert/kernels/PrimKernelRegistry.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/nativert/kernels`):

- [`PrimKernelRegistry.cpp_docs.md`](./PrimKernelRegistry.cpp_docs.md)
- [`AutoFunctionalizeKernel.cpp_docs.md`](./AutoFunctionalizeKernel.cpp_docs.md)
- [`ETCallDelegateKernel.cpp_docs.md`](./ETCallDelegateKernel.cpp_docs.md)
- [`HigherOrderKernel.cpp_docs.md`](./HigherOrderKernel.cpp_docs.md)
- [`KernelHandlerRegistry.cpp_docs.md`](./KernelHandlerRegistry.cpp_docs.md)
- [`NativeKernels.cpp_docs.md`](./NativeKernels.cpp_docs.md)
- [`KernelFactory.cpp_docs.md`](./KernelFactory.cpp_docs.md)
- [`AutoFunctionalizeKernel.h_docs.md`](./AutoFunctionalizeKernel.h_docs.md)
- [`HigherOrderKernel.h_docs.md`](./HigherOrderKernel.h_docs.md)


## Cross-References

- **File Documentation**: `KernelRegistry.h_docs.md`
- **Keyword Index**: `KernelRegistry.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/kernels`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/kernels`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/nativert/kernels`):

- [`ETCallDelegateKernel.cpp_docs.md_docs.md`](./ETCallDelegateKernel.cpp_docs.md_docs.md)
- [`TritonKernel.h_kw.md_docs.md`](./TritonKernel.h_kw.md_docs.md)
- [`PrimKernelRegistry.cpp_kw.md_docs.md`](./PrimKernelRegistry.cpp_kw.md_docs.md)
- [`C10Kernel.h_kw.md_docs.md`](./C10Kernel.h_kw.md_docs.md)
- [`CallTorchBindKernel.cpp_docs.md_docs.md`](./CallTorchBindKernel.cpp_docs.md_docs.md)
- [`ETCallDelegateKernel.h_kw.md_docs.md`](./ETCallDelegateKernel.h_kw.md_docs.md)
- [`AutoFunctionalizeKernel.h_kw.md_docs.md`](./AutoFunctionalizeKernel.h_kw.md_docs.md)
- [`HigherOrderKernel.h_docs.md_docs.md`](./HigherOrderKernel.h_docs.md_docs.md)
- [`C10Kernel.h_docs.md_docs.md`](./C10Kernel.h_docs.md_docs.md)
- [`CallTorchBindKernel.h_kw.md_docs.md`](./CallTorchBindKernel.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `KernelRegistry.h_docs.md_docs.md`
- **Keyword Index**: `KernelRegistry.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
