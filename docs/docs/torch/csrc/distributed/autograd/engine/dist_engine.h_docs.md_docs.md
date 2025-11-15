# Documentation: `docs/torch/csrc/distributed/autograd/engine/dist_engine.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/autograd/engine/dist_engine.h_docs.md`
- **Size**: 9,674 bytes (9.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/autograd/engine/dist_engine.h`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/engine/dist_engine.h`
- **Size**: 7,418 bytes (7.24 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <mutex>
#include <unordered_set>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/distributed/autograd/context/context.h>

namespace torch::distributed::autograd {

// Forward declaration.
class BackwardPassCleanupGuard;

// This is a singleton class responsible for running distributed backward
// passes. This engine relies heavily on the vanilla autograd engine and tries
// to reuse it as much as possible. This class is mostly responsible for the
// distributed aspects of autograd and tries to hook into the autograd engine
// where convenient.

// Unlike the vanilla autograd engine, the distributed autograd engine
// accumulates the gradients in the appropriate DistAutogradContext. This avoids
// multiple trainer nodes stomping on each others gradients.
class TORCH_API DistEngine {
 public:
  // Retrieve the singleton instance.
  static DistEngine& getInstance();

  // Given a list of root variables, start the distributed backwards pass from
  // these variables and accumulate all the gradients in the current autograd
  // context on each node. This method is used to kickoff distributed autograd
  // on a single node.
  void execute(
      int64_t context_id,
      const torch::autograd::variable_list& roots,
      bool retainGraph);

  // Given a send function to execute in the autograd engine, ensures we compute
  // dependencies once for this node and enqueues the send function for execute
  // in the engine.
  // This method is used to kick off the autograd computation on a node when it
  // receives gradients from the corresponding 'recv' method on another node.
  // The gradients are accumulated in the provided autograd context.
  c10::intrusive_ptr<c10::ivalue::Future> executeSendFunctionAsync(
      const ContextPtr& autogradContext,
      const std::shared_ptr<SendRpcBackward>& sendFunction,
      bool retainGraph);

  // Number of backward passes currently running for the Distributed Engine.
  size_t numBackwardPasses() const;

  // Returns key-value pairs consisting of useful debugging information related
  // to distributed autograd.
  std::unordered_map<std::string, int64_t> getDebugInfo() const;

  DistEngine(const DistEngine&) = delete;
  DistEngine& operator=(const DistEngine&) = delete;
  DistEngine(DistEngine&&) = delete;
  DistEngine& operator=(DistEngine&&) = delete;

 private:
  // Make sure this is a singleton.
  DistEngine();
  ~DistEngine();

  // Validates the input roots for the backward computations and retrieves the
  // appropriate root edges and corresponding gradients. Populates root_edges
  // with the appropriate gradient edges and grads with the gradients for each
  // edge.
  void validateRootsAndRetrieveEdges(
      const torch::autograd::variable_list& roots,
      torch::autograd::edge_list& rootEdges,
      torch::autograd::variable_list& grads);

  // Given the autograd context, root edges and grads, we compute dependencies
  // for the local node and fill out the provided GraphTask and GraphRoot with
  // appropriate information for the local autograd engine.
  // We also determine all leaf nodes(functions) in the graph and accumulate
  // them in outputEdges.
  void computeDependencies(
      const ContextPtr& context,
      const torch::autograd::edge_list& rootEdges,
      const torch::autograd::variable_list& grads,
      const std::shared_ptr<torch::autograd::Node>& graphRoot,
      torch::autograd::edge_list& outputEdges,
      bool retainGraph);

  // Given a pre-populated GraphTask and a root node, compute the backward pass
  // for the autograd graph until the graph task ready queue is empty.
  //
  // This method assumes that the appropriate GraphTask has already been
  // initialized appropriately. It will construct a local ready queue to
  // traverse the GraphTask instead of using the GraphTask embedded
  // cpu_ready_queue, this is because dist engine might run the same GraphTask
  // from different SendFunctions concurrently in different threads. The method
  // will only mark the GraphTask as completed when it needs to, which means it
  // might not mark as completed for every call as dist engine would like to
  // keep the GraphTask alive when it not receives all gradients.
  //
  // When `incrementOutstandingTasks=false`, the function does not increment
  // 'outstanding_tasks_' in the appropriate GraphTask. It is assumed we've
  // already done this before hand for this task (to ensure we don't pre-mark
  // this graph_task as completed). This is useful in the distributed autograd
  // case where we need to increment 'outstanding_tasks_' first to indicate the
  // local autograd engine the graph task is not completed until it receives the
  // signals from other workers over the network.
  //
  // XXX: calling this function assumes that we will have NO GPU nodetasks be
  // executed for the graph_task, the caller of this function need to ensure
  // this otherwise there will be undefined behaviors. A correct way to fix this
  // is to re-design the autograd engine so that GPU worker thread to behave the
  // same as CPU caller thread, record the operation/thread for the device, and
  // reuse it in backward.
  // TODO: 1. Add assert in the dist engine to ensure no GPU NodeTasks during
  // backward
  //       2. properly setup the thread local ready queue to enable reentrant
  //       backwards
  void execute_graph_task_until_ready_queue_empty(
      torch::autograd::NodeTask&& node_task,
      bool incrementOutstandingTasks = true);

  // Run the local autograd engine using the provided graphTask and graphRoot
  // and accumulate the gradients part 'outputEdges' in the provided autograd
  // context.
  c10::intrusive_ptr<c10::ivalue::Future> runEngineAndAccumulateGradients(
      const ContextPtr& autogradContext,
      const std::shared_ptr<torch::autograd::Node>& graphRoot,
      const torch::autograd::edge_list& outputEdges,
      bool incrementOutStandingTasks = true);

  // Run after the backward pass is done to appropriately cleanup structures.
  void cleanupBackwardPass(const ContextPtr& autogradContext);

  // Global thread to execute CPU continuations.
  void globalCpuThread(
      const std::shared_ptr<torch::autograd::ReadyQueue>& ready_queue);

  // Set of autograd context_ids, which we have already initialized for
  // distributed autograd on this node (e.g.: already computed dependencies)
  std::unordered_set<int64_t> initializedContextIds_;

  mutable std::mutex initializedContextIdsLock_;

  // Reference to local autograd engine.
  torch::autograd::Engine& engine_;

  // Ready queue used by the CPU thread in distributed engine.
  // See Note [GPU to CPU continuations]
  std::shared_ptr<torch::autograd::ReadyQueue> global_cpu_ready_queue_;

  // See Note [GPU to CPU continuations]
  std::thread global_cpu_thread_;

  friend class BackwardPassCleanupGuard;
};

// Guard to clean up resources once the backward pass is done.
class BackwardPassCleanupGuard {
 public:
  explicit BackwardPassCleanupGuard(ContextPtr autogradContext)
      : autogradContext_(std::move(autogradContext)) {}

  ~BackwardPassCleanupGuard() {
    DistEngine::getInstance().cleanupBackwardPass(autogradContext_);
  }

 private:
  ContextPtr autogradContext_;
};

} // namespace torch::distributed::autograd

```



## High-Level Overview


This C++ file contains approximately 6 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `BackwardPassCleanupGuard`, `responsible`, `is`, `TORCH_API`, `a`, `BackwardPassCleanupGuard`, `BackwardPassCleanupGuard`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/autograd/engine`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `mutex`
- `unordered_set`
- `torch/csrc/autograd/engine.h`
- `torch/csrc/autograd/function.h`
- `torch/csrc/autograd/functions/basic_ops.h`
- `torch/csrc/distributed/autograd/context/context.h`


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

Files in the same folder (`torch/csrc/distributed/autograd/engine`):

- [`dist_engine.cpp_docs.md`](./dist_engine.cpp_docs.md)


## Cross-References

- **File Documentation**: `dist_engine.h_docs.md`
- **Keyword Index**: `dist_engine.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/autograd/engine`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/autograd/engine`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed/autograd/engine`):

- [`dist_engine.h_kw.md_docs.md`](./dist_engine.h_kw.md_docs.md)
- [`dist_engine.cpp_kw.md_docs.md`](./dist_engine.cpp_kw.md_docs.md)
- [`dist_engine.cpp_docs.md_docs.md`](./dist_engine.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `dist_engine.h_docs.md_docs.md`
- **Keyword Index**: `dist_engine.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
