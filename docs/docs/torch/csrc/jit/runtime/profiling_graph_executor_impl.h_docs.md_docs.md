# Documentation: `docs/torch/csrc/jit/runtime/profiling_graph_executor_impl.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/profiling_graph_executor_impl.h_docs.md`
- **Size**: 5,662 bytes (5.53 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/profiling_graph_executor_impl.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/profiling_graph_executor_impl.h`
- **Size**: 2,958 bytes (2.89 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <c10/util/Flags.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

TORCH_DECLARE_bool(torch_jit_static_then_dynamic);

TORCH_DECLARE_bool(torch_jit_always_dynamic);

C10_DECLARE_bool(torch_jit_release_profiling_graph_after_optimization);
C10_DECLARE_int32(torch_jit_release_profiling_graph_delay_in_seconds);
C10_DECLARE_int64(torch_jit_num_profiled_runs);
C10_DECLARE_int64(torch_jit_bailout_depth);

namespace torch::jit {

TORCH_API void runNooptPassPipeline(std::shared_ptr<Graph>& graph);

struct TORCH_API ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  ProfilingGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);

  const ExecutionPlan& getPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth) override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;

  void debugFlushCompilationCache();

  bool isOptimized() const override {
    return optimized_plan_.has_value();
  }

 private:
  const ExecutionPlan& getOptimizedPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth);
  void runProfilingInsensitiveOptimizations(std::shared_ptr<Graph>& graph);
  void runProfilingOptimizations(
      std::shared_ptr<Graph>& graph,
      size_t remaining_depth);
  void replaceFallbackGraphWithFallbackFunction(Block* b);
  FusionBehavior getCurrentBehavior(size_t remaining_depth);
  size_t getInstantiatedBailoutDepth();
  void runNoGradOptimizations(
      std::shared_ptr<Graph>& graph,
      size_t remaining_bailout_depth);
  void runFinalOptimizations(std::shared_ptr<Graph>& graph);

  void clearTheGraphCompilationIntermediateGraphs();

  std::unique_ptr<ProfilingRecord> pr_;
  std::optional<ExecutionPlan>
      profiling_plan_; // plan to run in order to profiling the code
  std::optional<ExecutionPlan> optimized_plan_;
  FusionStrategy fusion_strategy_;

  // this plan is used if getGraphExecutorOptimize is unset
  std::optional<ExecutionPlan> fallback_plan_;
  // fallback functions are inserted for tensorexpr fusion groups
  // and by specialize_autogradzero. Whenever, at runtime, input
  // tensor don't match profiled properties, fallback functions are called
  // They are the deoptimized version of the logic in fusion groups
  // and/or autograd.
  // The fallback functions are owned by a GraphExecutor instance
  // They only exist in the optimized graph which is a private property
  // of the GraphExecutor and only shared with InterpreterState
  std::vector<std::unique_ptr<Function>> fallback_functions_;
  std::optional<size_t> remaining_bailout_depth_;
  // The time the optimized_plan_ is created.
  int32_t time_optimized_plan_created_ = 0;
  // Has the extra memory used by the graph for profiling is released?
  bool is_graph_extra_memory_released_ = false;
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Flags.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/runtime/graph_executor_impl.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `profiling_graph_executor_impl.h_docs.md`
- **Keyword Index**: `profiling_graph_executor_impl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/csrc/jit/runtime`):

- [`register_ops_utils.h_docs.md_docs.md`](./register_ops_utils.h_docs.md_docs.md)
- [`register_c10_ops.cpp_docs.md_docs.md`](./register_c10_ops.cpp_docs.md_docs.md)
- [`exception_message.h_kw.md_docs.md`](./exception_message.h_kw.md_docs.md)
- [`register_prim_ops.cpp_kw.md_docs.md`](./register_prim_ops.cpp_kw.md_docs.md)
- [`autodiff.cpp_kw.md_docs.md`](./autodiff.cpp_kw.md_docs.md)
- [`decomposition_registry_util.h_docs.md_docs.md`](./decomposition_registry_util.h_docs.md_docs.md)
- [`slice_indices_adjust.cpp_docs.md_docs.md`](./slice_indices_adjust.cpp_docs.md_docs.md)
- [`graph_iterator.h_kw.md_docs.md`](./graph_iterator.h_kw.md_docs.md)
- [`shape_function_registry.h_docs.md_docs.md`](./shape_function_registry.h_docs.md_docs.md)
- [`symbolic_script.cpp_docs.md_docs.md`](./symbolic_script.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `profiling_graph_executor_impl.h_docs.md_docs.md`
- **Keyword Index**: `profiling_graph_executor_impl.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
