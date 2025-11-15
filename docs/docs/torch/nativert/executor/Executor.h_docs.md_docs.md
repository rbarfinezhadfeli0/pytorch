# Documentation: `docs/torch/nativert/executor/Executor.h_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/executor/Executor.h_docs.md`
- **Size**: 8,993 bytes (8.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/executor/Executor.h`

## File Metadata

- **Path**: `torch/nativert/executor/Executor.h`
- **Size**: 5,906 bytes (5.77 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <atomic>
#include <memory>

#include <c10/util/FbcodeMaps.h>
#include <c10/util/Logging.h>
#include <c10/util/Semaphore.h>
#include <c10/util/Synchronized.h>

#include <torch/nativert/detail/ITree.h>
#include <torch/nativert/detail/MPMCQueue.h>
#include <torch/nativert/executor/ConstantFolder.h>
#include <torch/nativert/executor/DelegateExecutor.h>
#include <torch/nativert/executor/ExecutionPlanner.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/GraphExecutorBase.h>
#include <torch/nativert/executor/memory/FunctionSchema.h>
#include <torch/nativert/executor/memory/LayoutPlanner.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/GraphSignature.h>
#include <torch/nativert/kernels/KernelFactory.h>

namespace torch::nativert {

using namespace torch::nativert::detail;

struct DistributedRunConfig;

/**
 * A very dumb executor. Basically just runs each node in order and contains a
 * giant unordered map for every intermediate, no optimizations applied.
 */
class Executor {
  class ExecutorFrameDeleter {
   public:
    explicit ExecutorFrameDeleter(Executor& e) : e_(&e) {}
    ExecutorFrameDeleter(ExecutorFrameDeleter&&) = default;
    ExecutorFrameDeleter& operator=(ExecutorFrameDeleter&&) = default;
    ExecutorFrameDeleter(const ExecutorFrameDeleter&) = default;
    ExecutorFrameDeleter& operator=(const ExecutorFrameDeleter&) = default;
    ~ExecutorFrameDeleter() = default;

    void operator()(ExecutionFrame* p) {
      e_->returnExecutorFrameToPool(std::unique_ptr<ExecutionFrame>(p));
    }

   private:
    Executor* e_;
  };
  class ExecutorFramePtr {
   public:
    ExecutorFramePtr(std::unique_ptr<ExecutionFrame> ptr, Executor& e)
        : ptr_(std::unique_ptr<ExecutionFrame, ExecutorFrameDeleter>(
              ptr.release(),
              ExecutorFrameDeleter{e})) {}
    ExecutorFramePtr() = delete;
    ExecutorFramePtr(ExecutorFramePtr&&) = default;
    ExecutorFramePtr& operator=(ExecutorFramePtr&&) = default;
    ExecutorFramePtr(const ExecutorFramePtr&) = delete;
    ExecutorFramePtr& operator=(const ExecutorFramePtr&) = delete;
    ~ExecutorFramePtr() = default;

    ExecutionFrame& operator*() {
      return *ptr_;
    }

    ExecutionFrame* operator->() {
      return ptr_.get();
    }

   private:
    std::unique_ptr<ExecutionFrame, ExecutorFrameDeleter> ptr_;
  };

 public:
  // Constructor used for Inference Path
  Executor(
      torch::nativert::ExecutorConfig executorConfig,
      std::shared_ptr<Graph> graph,
      const std::shared_ptr<Weights>& weights,
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          pytorchStreamReader = nullptr);

  std::shared_ptr<Weights> getWeights() {
    std::shared_ptr<Weights> ret;
    weights_.withLock([&](auto& w) { ret = w; });
    return ret;
  }

  void processWeights(const std::shared_ptr<Weights>& weights);
  void atomicSwapWeights(std::shared_ptr<Weights> weights);

  // This API only returns the flattened UserOutputs,
  // intended to be used for Inference path
  // TODO Investigate whether we should remove this, still seems
  //      useful for testing.
  std::vector<c10::IValue> execute(std::vector<c10::IValue> inputs);

  std::vector<c10::IValue> execute(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const ITreeSpec& inputTreeSpec);

  ProfileMetrics benchmarkIndividualNodes(
      const std::vector<std::vector<c10::IValue>>& inputsList,
      const uint32_t warmupRuns,
      const uint32_t mainRuns);

  const torch::nativert::GraphSignature& graphSignature() const {
    return graph_->signature();
  }

  static std::string className() {
    return "Executor.v0";
  }

  const torch::nativert::ExecutorConfig& executorConfig() const {
    return executorConfig_;
  }

  std::vector<DelegateExecutor*> getDelegates();

  // Get the number of execution frames in the pool
  auto getNumExecutionFrames() const {
    return numExecutionFrames_.load();
  }

  static c10::FastMap<std::string /* target */, torch::nativert::FunctionSchema>
  getKernelSchemas(const std::vector<std::unique_ptr<OpKernel>>& kernels);

 protected:
  torch::nativert::ExecutorConfig executorConfig_;

  std::shared_ptr<Graph> graph_;

  // manages the parameters, buffers and tensor constants
  c10::Synchronized<std::shared_ptr<Weights>> weights_;

  void initialize(
      const std::shared_ptr<Weights>& weights,
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          pytorchStreamReader);

  ExecutorFramePtr getExecutorFrameFromPool();
  void returnExecutorFrameToPool(std::unique_ptr<ExecutionFrame> frame);

  // Clears stale execution frames from the pool
  void clearStaleExecutionFrames();

 private:
  void maybeRunConstantFolding(const std::shared_ptr<Weights>& weights);
  void validateInputs(const std::vector<c10::IValue>& inputs) const;

  // Helper method to get current timestamp in seconds
  int64_t getCurrentTimestampSeconds() const;

  void initWeights(const std::shared_ptr<Weights>& weights);

  std::unique_ptr<GraphExecutorBase> graphExecutor_;

  // NOTE: delegateExecutors_ is used by nodeKernels_ inside graphExecutor_.
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors_;

  std::vector<ConstFoldingExecution> constFoldingExecutions_;

  std::optional<ConstantFolder> constantFolder_;

  c10::Semaphore sem_;
  torch::nativert::detail::MPMCQueue<std::unique_ptr<ExecutionFrame>>
      executionFrames_;
  torch::nativert::detail::MPMCQueue<std::unique_ptr<ExecutionFrame>>
      inactiveExecutionFrames_;
  std::atomic_int64_t numExecutionFrames_;

  std::unique_ptr<LayoutPlanner> layoutPlanner_;
  std::atomic_int64_t lastClearedTimestamp_;
  std::mutex cleanupLock_;
  std::atomic_bool clearingInProgress_{false};
};

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `DistributedRunConfig`, `Executor`, `ExecutorFrameDeleter`, `ExecutorFramePtr`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/executor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `atomic`
- `memory`
- `c10/util/FbcodeMaps.h`
- `c10/util/Logging.h`
- `c10/util/Semaphore.h`
- `c10/util/Synchronized.h`
- `torch/nativert/detail/ITree.h`
- `torch/nativert/detail/MPMCQueue.h`
- `torch/nativert/executor/ConstantFolder.h`
- `torch/nativert/executor/DelegateExecutor.h`
- `torch/nativert/executor/ExecutionPlanner.h`
- `torch/nativert/executor/ExecutorConfig.h`
- `torch/nativert/executor/GraphExecutorBase.h`
- `torch/nativert/executor/memory/FunctionSchema.h`
- `torch/nativert/executor/memory/LayoutPlanner.h`
- `torch/nativert/graph/Graph.h`
- `torch/nativert/graph/GraphSignature.h`
- `torch/nativert/kernels/KernelFactory.h`


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


## Cross-References

- **File Documentation**: `Executor.h_docs.md`
- **Keyword Index**: `Executor.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/executor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/executor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/nativert/executor`):

- [`GraphExecutorBase.h_docs.md_docs.md`](./GraphExecutorBase.h_docs.md_docs.md)
- [`Placement.h_kw.md_docs.md`](./Placement.h_kw.md_docs.md)
- [`ParallelGraphExecutor.cpp_kw.md_docs.md`](./ParallelGraphExecutor.cpp_kw.md_docs.md)
- [`ExecutionFrame.h_docs.md_docs.md`](./ExecutionFrame.h_docs.md_docs.md)
- [`Executor.cpp_kw.md_docs.md`](./Executor.cpp_kw.md_docs.md)
- [`SerialGraphExecutor.h_docs.md_docs.md`](./SerialGraphExecutor.h_docs.md_docs.md)
- [`AOTInductorModelContainerCudaShim.cpp_docs.md_docs.md`](./AOTInductorModelContainerCudaShim.cpp_docs.md_docs.md)
- [`ParallelGraphExecutor.h_docs.md_docs.md`](./ParallelGraphExecutor.h_docs.md_docs.md)
- [`Placement.cpp_kw.md_docs.md`](./Placement.cpp_kw.md_docs.md)
- [`DelegateExecutor.h_docs.md_docs.md`](./DelegateExecutor.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Executor.h_docs.md_docs.md`
- **Keyword Index**: `Executor.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
