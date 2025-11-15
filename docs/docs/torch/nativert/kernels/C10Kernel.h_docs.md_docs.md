# Documentation: `docs/torch/nativert/kernels/C10Kernel.h_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/kernels/C10Kernel.h_docs.md`
- **Size**: 5,544 bytes (5.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/kernels/C10Kernel.h`

## File Metadata

- **Path**: `torch/nativert/kernels/C10Kernel.h`
- **Size**: 2,749 bytes (2.68 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <c10/core/Device.h>
#include <torch/nativert/executor/memory/FunctionSchema.h>

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

// Implementation of Kernel for ATen operators
//
// This class exists to amortize per-kernel overhead by computing things during
// initialization instead of on every execution. Right now we are only
// amortizing schema resolution, and static arguments parsing,
// but in the future this could be extended to avoid operator dispatch and
// do better "Register" allocation (e.g. convert input/outputs to directly
// array accesses onto a set of registers, in concert with memory planning)
class C10Kernel : public OpKernel {
 public:
  C10Kernel() = delete; // deleted default constructor
  C10Kernel(
      const Node* node,
      OpKernelKind kind = OpKernelKind::kInterpreterFallbackKernel,
      AliasingSpec&& aliasingSpec = {});
  ~C10Kernel() override = default;

  [[nodiscard]] const c10::IValue& input(
      uint32_t i,
      ExecutionFrame& executionFrame) const override {
    if (Value* dynamicArg = arguments_.findDynamic(i)) {
      return executionFrame.getIValue(dynamicArg->id());
    }
    return attribute(i);
  }

  [[nodiscard]] const c10::IValue& attribute(uint32_t i) const {
    return arguments_.getStatic(i);
  }

  C10_ALWAYS_INLINE const FunctionSchema& schema() const {
    return schema_;
  }

  void computeInternal(ExecutionFrame& executionFrame) const override;

 private:
  c10::OperatorHandle op_;
  FunctionSchema schema_;

  Arguments arguments_;
};

class SymIntOpKernel : public OpKernel {
 public:
  explicit SymIntOpKernel(const Node* node) : OpKernel(node) {}
  void computeInternal(ExecutionFrame& executionFrame) const final;
};

class SymBoolOpKernel : public OpKernel {
 public:
  explicit SymBoolOpKernel(const Node* node) : OpKernel(node) {}
  void computeInternal(ExecutionFrame& executionFrame) const final;
};

class SymFloatOpKernel : public OpKernel {
 public:
  explicit SymFloatOpKernel(const Node* node) : OpKernel(node) {}
  void computeInternal(ExecutionFrame& executionFrame) const final;
};

// ScalarOpKernel does binary arithmetic operations on scalar values.
// Integers and floats are supported as input types. The output will be
// promoted to float if and only if there's at least one float input.
class ScalarBinaryOpKernel : public OpKernel {
 public:
  explicit ScalarBinaryOpKernel(const Node* node) : OpKernel(node) {}
  void computeInternal(ExecutionFrame& executionFrame) const final;
};

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 6 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `exists`, `C10Kernel`, `SymIntOpKernel`, `SymBoolOpKernel`, `SymFloatOpKernel`, `ScalarBinaryOpKernel`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/dispatch/Dispatcher.h`
- `ATen/core/function_schema.h`
- `c10/core/Device.h`
- `torch/nativert/executor/memory/FunctionSchema.h`
- `torch/nativert/executor/ExecutionFrame.h`
- `torch/nativert/executor/OpKernel.h`
- `torch/nativert/graph/Graph.h`


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
- [`KernelRegistry.h_docs.md`](./KernelRegistry.h_docs.md)
- [`AutoFunctionalizeKernel.cpp_docs.md`](./AutoFunctionalizeKernel.cpp_docs.md)
- [`ETCallDelegateKernel.cpp_docs.md`](./ETCallDelegateKernel.cpp_docs.md)
- [`HigherOrderKernel.cpp_docs.md`](./HigherOrderKernel.cpp_docs.md)
- [`KernelHandlerRegistry.cpp_docs.md`](./KernelHandlerRegistry.cpp_docs.md)
- [`NativeKernels.cpp_docs.md`](./NativeKernels.cpp_docs.md)
- [`KernelFactory.cpp_docs.md`](./KernelFactory.cpp_docs.md)
- [`AutoFunctionalizeKernel.h_docs.md`](./AutoFunctionalizeKernel.h_docs.md)
- [`HigherOrderKernel.h_docs.md`](./HigherOrderKernel.h_docs.md)


## Cross-References

- **File Documentation**: `C10Kernel.h_docs.md`
- **Keyword Index**: `C10Kernel.h_kw.md`
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
- [`CallTorchBindKernel.h_kw.md_docs.md`](./CallTorchBindKernel.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `C10Kernel.h_docs.md_docs.md`
- **Keyword Index**: `C10Kernel.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
