# Documentation: `torch/csrc/jit/runtime/graph_executor_impl.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/graph_executor_impl.h`
- **Size**: 4,027 bytes (3.93 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/runtime/logging.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch::jit {

void packGradient(const Gradient& gradient, Node* dnode);
bool needsGradient(const std::shared_ptr<const Graph>& graph);
void runOptimization(
    std::shared_ptr<Graph>& graph,
    bool unroll_non_constant_loops = true,
    bool const_prop_user_classes = true);
void runNondiffOptimization(
    std::shared_ptr<Graph>& graph,
    bool strict_fuser_check = false);
void debugSetAutodiffSubgraphInlining(bool state);
bool TORCH_API getAutodiffSubgraphInlining();

void debugSetFusionGroupInlining(bool state);
bool getFusionGroupInlining();

// Tunable parameters for deciding when to create/keep subgraphs of
// differentiable code
const size_t autodiffSubgraphNodeThreshold = 2;
const size_t autodiffSubgraphInlineThreshold = 5;

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each
// situation. GraphExecutor is completely unaware of tracing or module
// parameters to keep the tracing concerns separated.
struct GraphExecutorImplBase {
  static std::shared_ptr<Graph> prepareGraph(
      const std::shared_ptr<Graph>& graph) {
    auto copy = graph->copy();
    EraseShapeInformation(copy);
    return copy;
  }

  GraphExecutorImplBase(
      const std::shared_ptr<Graph>& graph,
      std::string function_name)
      : graph(prepareGraph(graph)),
        function_name_(std::move(function_name)),
        num_inputs(this->graph->inputs().size()),
        num_outputs(this->graph->outputs().size()) {}

  // entry point where execution begins
  void run(Stack& stack);
  c10::intrusive_ptr<Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch);

  virtual const ExecutionPlan& getPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth = std::nullopt) = 0;
  virtual GraphExecutorState getDebugState() = 0;
  virtual ~GraphExecutorImplBase() = default;

  virtual bool isOptimized() const {
    return false;
  }

 protected:
  friend struct GraphExecutor;

  // The unoptimized starting graph. This field is effectively const, but we
  // can't make it so because Graph::copy() is not const (and making it const is
  // not that easy at this point).
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Graph> graph;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::string function_name_;

  // If false, we'll run the graph as we get it, without any optimizations.
  // Useful for debugging.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const size_t num_inputs;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const size_t num_outputs;

  // GraphExecutors can be accessed from multiple threads, so this thread needs
  // to be held every time we access the fallback or plan_cache.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex compile_mutex;
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `GraphExecutorImplBase`, `GraphExecutor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/runtime/graph_executor.h`
- `ATen/core/ivalue.h`
- `c10/util/Exception.h`
- `torch/csrc/autograd/grad_mode.h`
- `torch/csrc/jit/frontend/tracer.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/passes/shape_analysis.h`
- `torch/csrc/jit/resource_guard.h`
- `torch/csrc/jit/runtime/argument_spec.h`
- `torch/csrc/jit/runtime/autodiff.h`
- `torch/csrc/jit/runtime/custom_operator.h`
- `torch/csrc/jit/runtime/interpreter.h`
- `torch/csrc/jit/runtime/profiling_record.h`
- `torch/csrc/autograd/edge.h`
- `torch/csrc/autograd/function.h`
- `torch/csrc/jit/frontend/ir_emitter.h`
- `torch/csrc/jit/runtime/logging.h`
- `cstdint`
- `iterator`
- `memory`
- `mutex`
- `unordered_map`
- `utility`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `graph_executor_impl.h_docs.md`
- **Keyword Index**: `graph_executor_impl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
