# Documentation: `torch/nativert/executor/GraphExecutorBase.cpp`

## File Metadata

- **Path**: `torch/nativert/executor/GraphExecutorBase.cpp`
- **Size**: 4,427 bytes (4.32 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/record_function.h>
#include <torch/nativert/executor/GraphExecutorBase.h>

#include <c10/util/Logging.h>
#include <caffe2/core/timer.h>

namespace torch::nativert {

GraphExecutorBase::GraphExecutorBase(
    const Graph& graph,
    std::vector<std::unique_ptr<OpKernel>> nodeKernels,
    const ExecutorConfig& executorConfig)
    : graph_(graph),
      nodeKernels_(std::move(nodeKernels)),
      executorConfig_(executorConfig),
      execPlan_(ExecutionPlanner{graph_}.createPlan()) {}

void GraphExecutorBase::fillUserInputs(
    ExecutionFrame& frame,
    std::vector<c10::IValue> inputs) {
  RECORD_USER_SCOPE("Executor::fillUserInputs");
  const auto& inputValues = graph_.userInputs();
  TORCH_CHECK(inputValues.size() == inputs.size());

  // load user input tensor into execution frame
  for (size_t i = 0; i < inputValues.size(); i++) {
    if (inputValues[i]) {
      frame.setIValue(inputValues[i]->id(), std::move(inputs[i]));
    }
  }
}

ProfileMetrics GraphExecutorBase::benchmarkIndividualNodes(
    ExecutionFrame& executionFrame,
    const std::vector<std::vector<c10::IValue>>& inputsList,
    const uint32_t warmupRuns,
    const uint32_t mainRuns) {
  // TODO: add support for memory profiling
  TORCH_CHECK(warmupRuns >= 1 && mainRuns >= 1);

  ProfileMetrics results;
  const auto numNodes = static_cast<uint32_t>(nodeKernels_.size());

  results.percentPerNode.resize(numNodes, 0.0f);
  results.nodeTypes.reserve(numNodes);
  for (const auto& nodeKernel : nodeKernels_) {
    results.nodeTypes.emplace_back(nodeKernel->node()->target());
  }

  results.timePerNode.resize(numNodes, 0);
  if (inputsList.empty()) {
    auto i = 0;
    for (const auto& nodeKernel : nodeKernels_) {
      std::string target(nodeKernel->node()->target());
      results.timePerNode[i] = 0;
      results.timePerNodeType[target] = 0;
      results.instancesPerNodeType[target]++;
      if (nodeKernel->hasPrimKernel()) {
        results.primNodesCount++;
        results.primNodes.insert(target);
      } else if (nodeKernel->hasStaticDispatch()) {
        results.staticDispatchNodesCount++;
        results.staticDispatchNodes.insert(target);
      }
      i++;
    }
    results.totalNodesCount = numNodes;
    for (const auto& p : results.timePerNodeType) {
      const std::string& kind = p.first;
      results.percentPerNodeType[kind] = 0;
    }
    return results;
  }

  // Warmup
  for (uint32_t i = 0; i < warmupRuns; i++) {
    for (const auto& inputs : inputsList) {
      execute(executionFrame, inputs);
    }
  }

  // Execute kernels
  caffe2::Timer timer;
  executionFrame.withManagedMemory([&](auto) {
    for (uint32_t i = 0; i < mainRuns; i++) {
      for (auto inputs : inputsList) {
        const auto& inputValues = graph_.userInputs();

        TORCH_CHECK(inputValues.size() == inputs.size());
        for (size_t j = 0; j < inputValues.size(); j++) {
          executionFrame.setIValue(inputValues[j]->id(), std::move(inputs[j]));
        }
        for (NodeIndex nodeIdx = 0; nodeIdx < nodeKernels_.size(); ++nodeIdx) {
          timer.Start();
          nodeKernels_[nodeIdx]->compute(executionFrame);
          float millis = timer.MilliSeconds();
          results.timePerNode[nodeIdx] += millis;
        }
      }
    }
  });

  // Summarize results
  const float numTotalIters =
      (static_cast<float>(mainRuns) * static_cast<float>(inputsList.size()));
  for (const auto i : c10::irange(numNodes)) {
    const Node* node = nodeKernels_[i]->node();
    std::string target(node->target());
    results.timePerNode[i] /= numTotalIters;
    results.timePerNodeType[target] += results.timePerNode[i];
    results.instancesPerNodeType[target]++;
    if (nodeKernels_[i]->hasPrimKernel()) {
      results.primNodes.insert(target);
      results.primNodesCount++;
    } else if (nodeKernels_[i]->hasStaticDispatch()) {
      results.staticDispatchNodes.insert(target);
      results.staticDispatchNodesCount++;
    }
    results.totalTime += results.timePerNode[i];
  }
  results.totalNodesCount = numNodes;
  for (const auto& r : results.timePerNodeType) {
    const std::string& target = r.first;
    results.percentPerNodeType[target] = r.second * 100.0f / results.totalTime;
  }
  for (const auto i : c10::irange(numNodes)) {
    results.percentPerNode[i] =
        results.timePerNode[i] * 100.0f / results.totalTime;
  }
  return results;
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/executor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/record_function.h`
- `torch/nativert/executor/GraphExecutorBase.h`
- `c10/util/Logging.h`
- `caffe2/core/timer.h`


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

- **File Documentation**: `GraphExecutorBase.cpp_docs.md`
- **Keyword Index**: `GraphExecutorBase.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
