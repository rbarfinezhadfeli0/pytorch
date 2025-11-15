# Documentation: `docs/torch/nativert/executor/ParallelGraphExecutor.h_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/executor/ParallelGraphExecutor.h_docs.md`
- **Size**: 5,328 bytes (5.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/executor/ParallelGraphExecutor.h`

## File Metadata

- **Path**: `torch/nativert/executor/ParallelGraphExecutor.h`
- **Size**: 2,621 bytes (2.56 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/Semaphore.h>
#include <torch/nativert/executor/GraphExecutorBase.h>
#include <torch/nativert/executor/SessionState.h>
#include <thread>

namespace moodycamel {
struct ProducerToken;
struct ConsumerToken;
struct ConcurrentQueueDefaultTraits;
template <typename T, typename Traits>
class ConcurrentQueue;
} // namespace moodycamel

namespace torch::nativert {
class ThreadPoolExecutor;

typedef std::function<void()> Work;

struct WorkUnit {
  const Node* node;
  OpKernel* kernel;
  std::vector<WorkUnit*> users;
  void run(ThreadPoolExecutor* executor, SessionState* sessionState);
};

class ThreadPoolExecutor {
 public:
  explicit ThreadPoolExecutor();
  ~ThreadPoolExecutor();
  ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
  ThreadPoolExecutor& operator=(ThreadPoolExecutor const&) = delete;
  ThreadPoolExecutor(ThreadPoolExecutor&&) = delete;
  ThreadPoolExecutor& operator=(ThreadPoolExecutor&&) = delete;

  void run(SessionState& session, const std::vector<WorkUnit*>& roots);

  void start(int32_t numThreads);
  void stop();

  // execute unit on the current thread
  // NOTE: children can still be offloaded to other threads
  C10_ALWAYS_INLINE void execute_inline(SessionState* session, WorkUnit* unit);

  void add(SessionState* session, WorkUnit* unit);
  void add(
      SessionState* session,
      std::vector<WorkUnit*>::const_iterator begin,
      const std::vector<WorkUnit*>::const_iterator& end);

  C10_ALWAYS_INLINE moodycamel::ProducerToken& ptok();
  C10_ALWAYS_INLINE moodycamel::ConsumerToken& ctok();

 private:
  void loop();

  std::atomic_bool stopped_{false};

  std::unique_ptr<c10::Semaphore> sem_{std::make_unique<c10::Semaphore>()};

  std::unique_ptr<moodycamel::ConcurrentQueue<
      Work,
      moodycamel::ConcurrentQueueDefaultTraits>>
      work_;
  std::vector<std::thread> threads_;
};

class ParallelGraphExecutor : public GraphExecutorBase {
 public:
  ParallelGraphExecutor(
      const Graph& graph,
      std::vector<std::unique_ptr<OpKernel>> nodeKernels,
      const ExecutorConfig& executorConfig);

  std::vector<c10::IValue> execute(
      ExecutionFrame& frame,
      std::vector<c10::IValue> inputs) override;

  std::vector<c10::IValue> executeWithPrefilledFrame(
      ExecutionFrame& frame) override;

 private:
  ThreadPoolExecutor executor_;

  std::vector<WorkUnit*> inputWorkUnits_;
  c10::FastMap<const Node*, WorkUnit*> nodeToWorkUnit_;
  std::vector<WorkUnit> workUnits_;

  const Graph& graph_;
  c10::FastMap<const Node*, copyable_atomic<std::uint_fast32_t>> producers_;
};

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `moodycamel`

**Classes/Structs**: `ProducerToken`, `ConsumerToken`, `ConcurrentQueueDefaultTraits`, `ConcurrentQueue`, `ThreadPoolExecutor`, `WorkUnit`, `ThreadPoolExecutor`, `ParallelGraphExecutor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/executor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Semaphore.h`
- `torch/nativert/executor/GraphExecutorBase.h`
- `torch/nativert/executor/SessionState.h`
- `thread`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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
- [`ExecutionFrame.h_docs.md`](./ExecutionFrame.h_docs.md)
- [`ExecutorConfig.h_docs.md`](./ExecutorConfig.h_docs.md)
- [`SerialGraphExecutor.h_docs.md`](./SerialGraphExecutor.h_docs.md)
- [`Weights.cpp_docs.md`](./Weights.cpp_docs.md)
- [`OpKernelKind.h_docs.md`](./OpKernelKind.h_docs.md)
- [`Executor.h_docs.md`](./Executor.h_docs.md)


## Cross-References

- **File Documentation**: `ParallelGraphExecutor.h_docs.md`
- **Keyword Index**: `ParallelGraphExecutor.h_kw.md`
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
- [`Placement.cpp_kw.md_docs.md`](./Placement.cpp_kw.md_docs.md)
- [`DelegateExecutor.h_docs.md_docs.md`](./DelegateExecutor.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ParallelGraphExecutor.h_docs.md_docs.md`
- **Keyword Index**: `ParallelGraphExecutor.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
