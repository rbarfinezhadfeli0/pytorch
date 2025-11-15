# Documentation: `torch/nativert/executor/GraphExecutorBase.h`

## File Metadata

- **Path**: `torch/nativert/executor/GraphExecutorBase.h`
- **Size**: 2,595 bytes (2.53 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/ExecutionPlanner.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/GraphSignature.h>

namespace torch::nativert {

struct ProfileMetrics {
  size_t primNodesCount{0};
  size_t staticDispatchNodesCount{0};
  size_t totalNodesCount{0};
  std::vector<float> timePerNode;
  std::vector<std::string> nodeTypes;
  std::unordered_map<std::string, float> timePerNodeType;
  std::unordered_map<std::string, float> percentPerNodeType;
  std::vector<float> percentPerNode;
  std::unordered_map<std::string, int> instancesPerNodeType;
  std::unordered_set<std::string> staticDispatchNodes;
  std::unordered_set<std::string> primNodes;
  float totalTime{0};
  std::string name;
};

/**
 * GraphExecutor is a lightweight abstraction to execute a graph with
 * execution frames without actually owning the graph nor the weights. This is
 * introduced to decouple the state management of the top level runtime from the
 * kernel executions so that sub graphs from higher order ops can be supported.
 */
class GraphExecutorBase {
 public:
  GraphExecutorBase(
      const Graph& graph,
      std::vector<std::unique_ptr<OpKernel>> nodeKernels,
      const ExecutorConfig& executorConfig);
  virtual ~GraphExecutorBase() = default;

  const Graph& graph() const {
    return graph_;
  }

  // This API only returns the flattened UserOutputs,
  // intended to be used for Inference path
  virtual std::vector<c10::IValue> execute(
      ExecutionFrame& frame,
      std::vector<c10::IValue> inputs) = 0;

  virtual std::vector<c10::IValue> executeWithPrefilledFrame(
      ExecutionFrame& frame) = 0;

  ProfileMetrics benchmarkIndividualNodes(
      ExecutionFrame& executionFrame,
      const std::vector<std::vector<c10::IValue>>& inputs,
      const uint32_t warmup_runs,
      const uint32_t main_runs);

  std::vector<std::unique_ptr<OpKernel>> stealKernels() {
    return std::move(nodeKernels_);
  }

  void setKernels(std::vector<std::unique_ptr<OpKernel>>&& kernels) {
    nodeKernels_ = std::move(kernels);
  }

 protected:
  void fillUserInputs(ExecutionFrame& frame, std::vector<c10::IValue> inputs);

  const Graph& graph_;

  // cache of the constructed kernels to avoid reconstruction per execution
  std::vector<std::unique_ptr<OpKernel>> nodeKernels_;

  const ExecutorConfig& executorConfig_;

  std::unique_ptr<ExecutionPlan> execPlan_;
};

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ProfileMetrics`, `GraphExecutorBase`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/executor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nativert/executor/ExecutionFrame.h`
- `torch/nativert/executor/ExecutionPlanner.h`
- `torch/nativert/executor/ExecutorConfig.h`
- `torch/nativert/executor/OpKernel.h`
- `torch/nativert/graph/Graph.h`
- `torch/nativert/graph/GraphSignature.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/nativert/executor`):

- [`OpKernel.cpp_docs.md`](./OpKernel.cpp_docs.md)
- [`AOTInductorDelegateExecutor.cpp_docs.md`](./AOTInductorDelegateExecutor.cpp_docs.md)
- [`ExecutionFrame.cpp_docs.md`](./ExecutionFrame.cpp_docs.md)
- [`ParallelGraphExecutor.h_docs.md`](./ParallelGraphExecutor.h_docs.md)
- [`ExecutionFrame.h_docs.md`](./ExecutionFrame.h_docs.md)
- [`ExecutorConfig.h_docs.md`](./ExecutorConfig.h_docs.md)
- [`SerialGraphExecutor.h_docs.md`](./SerialGraphExecutor.h_docs.md)
- [`Weights.cpp_docs.md`](./Weights.cpp_docs.md)
- [`OpKernelKind.h_docs.md`](./OpKernelKind.h_docs.md)
- [`Executor.h_docs.md`](./Executor.h_docs.md)


## Cross-References

- **File Documentation**: `GraphExecutorBase.h_docs.md`
- **Keyword Index**: `GraphExecutorBase.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
