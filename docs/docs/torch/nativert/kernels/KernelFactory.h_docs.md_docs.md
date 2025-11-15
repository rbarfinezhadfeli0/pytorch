# Documentation: `docs/torch/nativert/kernels/KernelFactory.h_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/kernels/KernelFactory.h_docs.md`
- **Size**: 5,472 bytes (5.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/kernels/KernelFactory.h`

## File Metadata

- **Path**: `torch/nativert/kernels/KernelFactory.h`
- **Size**: 2,697 bytes (2.63 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <memory>

#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>
#include <torch/nativert/executor/DelegateExecutor.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/GraphExecutorBase.h>
#include <torch/nativert/executor/OpKernel.h>

namespace torch::nativert {

struct ConstFoldingExecution {
  std::unique_ptr<GraphExecutorBase> executor;
};

struct ExecutionKernels {
  std::vector<std::unique_ptr<OpKernel>> nodeKernels;
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors;
  std::vector<ConstFoldingExecution> constFoldingExecutions;
};

class KernelFactoryHandler {
 public:
  using OpKernelPtr = std::unique_ptr<OpKernel>;
  using DelegateExecutorPtr = std::unique_ptr<DelegateExecutor>;
  using Matcher = c10::function_ref<
      bool(const Node& node, const torch::nativert::ExecutorConfig&)>;
  using Callback =
      c10::function_ref<std::pair<OpKernelPtr, DelegateExecutorPtr>(
          const Node&,
          std::shared_ptr<Weights> weights,
          const torch::nativert::ExecutorConfig& executorConfig,
          caffe2::serialize::PyTorchStreamReader* pytorchStreamReader)>;

  KernelFactoryHandler(Matcher matcher, Callback callback)
      : matcher_(matcher), callback_(callback) {}

  KernelFactoryHandler() = delete;
  KernelFactoryHandler(const KernelFactoryHandler&) = default;
  KernelFactoryHandler& operator=(const KernelFactoryHandler&) = default;
  KernelFactoryHandler(KernelFactoryHandler&&) = default;
  KernelFactoryHandler& operator=(KernelFactoryHandler&&) = default;
  ~KernelFactoryHandler() = default;

  bool match(const Node& node, const torch::nativert::ExecutorConfig& config)
      const {
    return matcher_(node, config);
  }

  std::pair<OpKernelPtr, DelegateExecutorPtr> operator()(
      const Node& node,
      std::shared_ptr<Weights> weights,
      const torch::nativert::ExecutorConfig& executorConfig,
      caffe2::serialize::PyTorchStreamReader* pytorchStreamReader) const {
    return callback_(node, weights, executorConfig, pytorchStreamReader);
  }

 private:
  Matcher matcher_;
  Callback callback_;
};

class KernelFactory {
 public:
  KernelFactory() = default;

  ExecutionKernels initializeNodeKernels(
      const Graph& graph,
      const std::shared_ptr<Weights>& weights,
      const torch::nativert::ExecutorConfig& executorConfig,
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          pytorchStreamReader = nullptr);

  static void registerHandler(
      const std::string& name,
      KernelFactoryHandler handler);

  static bool isHandlerRegistered(const std::string& handler);
};

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ConstFoldingExecution`, `ExecutionKernels`, `KernelFactoryHandler`, `KernelFactory`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `memory`
- `torch/csrc/inductor/aoti_torch/proxy_executor.h`
- `torch/nativert/executor/DelegateExecutor.h`
- `torch/nativert/executor/ExecutorConfig.h`
- `torch/nativert/executor/GraphExecutorBase.h`
- `torch/nativert/executor/OpKernel.h`


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

- **File Documentation**: `KernelFactory.h_docs.md`
- **Keyword Index**: `KernelFactory.h_kw.md`
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

- **File Documentation**: `KernelFactory.h_docs.md_docs.md`
- **Keyword Index**: `KernelFactory.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
