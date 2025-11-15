# Documentation: `torch/nativert/kernels/KernelHandlerRegistry.cpp`

## File Metadata

- **Path**: `torch/nativert/kernels/KernelHandlerRegistry.cpp`
- **Size**: 4,196 bytes (4.10 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nativert/kernels/KernelHandlerRegistry.h>

#include <c10/util/Logging.h>
#include <fmt/format.h>

#include <ATen/core/ivalue.h>
#include <c10/util/CallOnce.h>

#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/GraphPasses.h>
#include <torch/nativert/graph/GraphUtils.h>
#include <torch/nativert/kernels/KernelFactory.h>
#include <torch/nativert/kernels/KernelRegistry.h>

#include <torch/csrc/inductor/aoti_torch/oss_proxy_executor.h>
#include <torch/nativert/executor/AOTInductorDelegateExecutor.h>
#include <torch/nativert/kernels/ETCallDelegateKernel.h>

namespace torch::nativert {

namespace {
std::string maybeRevisedStaticDispatchTarget(const Node& node) {
  auto overloadName = selectScalarOverloadName(node);

  if (!overloadName.empty() && !c10::ends_with(node.target(), overloadName)) {
    const std::string& newTarget =
        std::string(node.target())
            .replace(node.target().rfind('.'), std::string::npos, overloadName);
    LOG(INFO) << fmt::format(
        "Converting Tensor to {} for node: {} -> {}",
        overloadName,
        node.target(),
        newTarget);
    return newTarget;
  }
  return std::string(node.target());
}

std::unique_ptr<torch::aot_inductor::ProxyExecutor> make_proxy_executor(
    const std::string& filename,
    bool is_cpu,
    std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs) {
  return std::make_unique<torch::aot_inductor::OSSProxyExecutor>(
      filename, is_cpu, std::move(custom_objs));
}
} // namespace

void register_kernel_handlers() {
  static c10::once_flag flag;
  c10::call_once(flag, []() {
    using OpKernelPtr = KernelFactoryHandler::OpKernelPtr;
    using DelegateExecutorPtr = KernelFactoryHandler::DelegateExecutorPtr;
    KernelFactory::registerHandler(
        "static_cpu",
        KernelFactoryHandler(
            [](const Node& node,
               const torch::nativert::ExecutorConfig& executorConfig) {
              if (!executorConfig.enableStaticCPUKernels ||
                  !torch::nativert::areAllIOTensorsAttributesOnCpu(node)) {
                return false;
              }
              const std::string target = maybeRevisedStaticDispatchTarget(node);
              return torch::nativert::StaticallyDispatchedCPUKernelRegistry()
                  ->Has(target);
            },
            [](const Node& node,
               // NOLINTNEXTLINE(performance-unnecessary-value-param)
               std::shared_ptr<Weights> weights,
               const torch::nativert::ExecutorConfig& executorConfig,
               caffe2::serialize::PyTorchStreamReader* packageReader)
                -> std::pair<OpKernelPtr, DelegateExecutorPtr> {
              return {
                  torch::nativert::StaticallyDispatchedCPUKernelRegistry()
                      ->Create(maybeRevisedStaticDispatchTarget(node), &node),
                  nullptr};
            }));
    KernelFactory::registerHandler(
        "et_delegate",
        KernelFactoryHandler(
            [](const Node& node,
               const torch::nativert::ExecutorConfig& /* executorConfig */) {
              return c10::starts_with(
                  node.target(),
                  "torch.ops.higher_order.executorch_call_delegate");
            },
            [](const Node& node,
               // NOLINTNEXTLINE(performance-unnecessary-value-param)
               std::shared_ptr<Weights> weights,
               const torch::nativert::ExecutorConfig& executorConfig,
               caffe2::serialize::PyTorchStreamReader* packageReader)
                -> std::pair<
                    KernelFactoryHandler::OpKernelPtr,
                    KernelFactoryHandler::DelegateExecutorPtr> {
              auto delegateExecutor = std::make_unique<AOTIDelegateExecutor>(
                  node,
                  weights,
                  executorConfig,
                  packageReader,
                  make_proxy_executor);

              return {
                  std::make_unique<ETCallDelegateKernel>(
                      &node, *delegateExecutor),
                  std::move(delegateExecutor)};
            }));
  });
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nativert/kernels/KernelHandlerRegistry.h`
- `c10/util/Logging.h`
- `fmt/format.h`
- `ATen/core/ivalue.h`
- `c10/util/CallOnce.h`
- `torch/nativert/graph/Graph.h`
- `torch/nativert/graph/GraphPasses.h`
- `torch/nativert/graph/GraphUtils.h`
- `torch/nativert/kernels/KernelFactory.h`
- `torch/nativert/kernels/KernelRegistry.h`
- `torch/csrc/inductor/aoti_torch/oss_proxy_executor.h`
- `torch/nativert/executor/AOTInductorDelegateExecutor.h`
- `torch/nativert/kernels/ETCallDelegateKernel.h`


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
- [`NativeKernels.cpp_docs.md`](./NativeKernels.cpp_docs.md)
- [`KernelFactory.cpp_docs.md`](./KernelFactory.cpp_docs.md)
- [`AutoFunctionalizeKernel.h_docs.md`](./AutoFunctionalizeKernel.h_docs.md)
- [`HigherOrderKernel.h_docs.md`](./HigherOrderKernel.h_docs.md)


## Cross-References

- **File Documentation**: `KernelHandlerRegistry.cpp_docs.md`
- **Keyword Index**: `KernelHandlerRegistry.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
