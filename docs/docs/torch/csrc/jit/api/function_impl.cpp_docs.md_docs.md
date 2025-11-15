# Documentation: `docs/torch/csrc/jit/api/function_impl.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/api/function_impl.cpp_docs.md`
- **Size**: 7,899 bytes (7.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/api/function_impl.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/api/function_impl.cpp`
- **Size**: 5,276 bytes (5.15 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/Flags.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/peephole.h>

#ifndef C10_MOBILE
#include <ATen/autocast_mode.h>
#include <torch/csrc/jit/passes/autocast.h>
#endif

// clang-format off
C10_DEFINE_bool(
    torch_jit_do_not_store_optimized_graph,
    false,
    "Do not store the optimized graph.")

namespace torch::jit {
namespace {
c10::FunctionSchema defaultSchemaFor(const GraphFunction& function) {
  std::vector<c10::Argument> args;
  std::vector<c10::Argument> returns;
  Graph& g = *function.graph();
  size_t num_inputs = function.num_inputs();
  for (const auto i : c10::irange(num_inputs)) {
    const Value* v = g.inputs().at(i);
    std::string name = v->hasDebugName() ? v->debugNameBase()
                                         : ("argument_" + std::to_string(i));
    args.emplace_back(std::move(name), unshapedType(g.inputs()[i]->type()));
  }
  for (const auto i : c10::irange(g.outputs().size())) {
    returns.emplace_back("", unshapedType(g.outputs()[i]->type()));
  }
  return {function.name(), "", std::move(args), std::move(returns)};
}

template <typename T, typename F>
T* tryToGraphFunctionImpl(F& function) noexcept {
  if (!function.isGraphFunction()) {
    return nullptr;
  }

  return static_cast<T*>(&function);
}

template <typename T, typename F>
T& toGraphFunctionImpl(F& function) {
  if (auto* g = tryToGraphFunctionImpl<T>(function)) {
    return *g;
  }

  TORCH_INTERNAL_ASSERT(
      false,
      "Failed to downcast a Function to a GraphFunction. "
      "This probably indicates that the JIT calling context needs a "
      "special case on tryToGraphFunction() instead.");
}

} // namespace

static void placeholderCreator(GraphFunction& /*unused*/) {
  throw RecursiveMethodCallError();
}

void GraphFunction::run(Stack& stack) {
  C10_LOG_EVENT_SAMPLED(run, qualname().qualifiedName(), stack);
  get_executor().run(stack);
}

c10::intrusive_ptr<c10::ivalue::Future> GraphFunction::runAsync(
    Stack& stack,
    TaskLauncher taskLauncher) {
  return get_executor().runAsync(stack, std::move(taskLauncher));
}

void GraphFunction::ensure_defined() {
  if (function_creator_) {
    auto creator = function_creator_;
    function_creator_ = placeholderCreator;
    creator(*this);
    function_creator_ = nullptr;
  }
  check_single_output();
}

const c10::FunctionSchema& GraphFunction::getSchema() const {
  if (schema_ == nullptr) {
    schema_ = std::make_unique<c10::FunctionSchema>(defaultSchemaFor(*this));
  }
  return *schema_;
}

std::shared_ptr<Graph> GraphFunction::optimized_graph() const {
  std::lock_guard<std::recursive_mutex> lock(compile_mutex);
  decltype(optimized_graphs_)::value_type graph;
  auto& graph_ref = !FLAGS_torch_jit_do_not_store_optimized_graph
      ? optimized_graphs_[currentSpecialization()]
      : graph;
  if (graph_ref) {
    return graph_ref;
  }
  graph_ref = graph_->copy();
  if (getGraphExecutorOptimize()) {
    preoptimizeGraph(graph_ref, force_no_amp_);
  }
  return graph_ref;
}

GraphFunction::SpecializationKey GraphFunction::currentSpecialization() const {
  if (force_no_amp_) {
    return SpecializationKey::AutocastOff;
  }
#ifdef C10_MOBILE
  // disabling autodiff pass for mobile build since autocast APIs don't exist
  return SpecializationKey::AutocastOff;
#else
  bool cpu_enabled = at::autocast::is_autocast_enabled(at::kCPU);
  bool gpu_enabled = at::autocast::is_autocast_enabled(at::kCUDA);
  if (cpu_enabled && gpu_enabled) {
    return SpecializationKey::CpuGpuAutocastOn;
  } else if (!cpu_enabled && !gpu_enabled) {
    return SpecializationKey::AutocastOff;
  } else {
    return gpu_enabled ? SpecializationKey::GpuAutocastOn
                       : SpecializationKey::CpuAutocastOn;
  }
#endif
}

void preoptimizeGraph(std::shared_ptr<Graph>& graph, bool disable_autocast) {
  Inline(*graph);

  // Peephole Optimize cleans up many "is None" checks and creates constant
  // prop opportunities
  PeepholeOptimize(graph, true);

  // AliasDb construction can be slow, so run it just on immutable types
  // to clean up constant Ifs & other easy wins
  ConstantPropagationImmutableTypes(graph);

#ifndef C10_MOBILE
  // Inject casts for automatic mixed precision
  //
  // TODO: Ideally, this pass could run earlier, before inlining
  //  or any other optimizations. That setup is preferable because:
  //  1. The AMP pass would be self-contained and function independently
  //     of the any optimizations
  //  2. AMP transformations would benefit from followup passes's cleanup
  //
  if (!disable_autocast) {
    Autocast(graph);
  }
#endif

  ConstantPooling(graph);
}

GraphFunction* tryToGraphFunction(Function& function) noexcept {
  return tryToGraphFunctionImpl<GraphFunction>(function);
}

GraphFunction& toGraphFunction(Function& function) {
  return toGraphFunctionImpl<GraphFunction>(function);
}

const GraphFunction& toGraphFunction(const Function& function) {
  return toGraphFunctionImpl<const GraphFunction>(function);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `static`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/api`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Flags.h`
- `c10/util/irange.h`
- `torch/csrc/jit/api/function_impl.h`
- `torch/csrc/jit/passes/inliner.h`
- `torch/csrc/jit/frontend/error_report.h`
- `torch/csrc/jit/passes/constant_pooling.h`
- `torch/csrc/jit/passes/constant_propagation.h`
- `torch/csrc/jit/passes/peephole.h`
- `ATen/autocast_mode.h`
- `torch/csrc/jit/passes/autocast.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/jit/api`):

- [`function_impl.h_docs.md`](./function_impl.h_docs.md)
- [`module.h_docs.md`](./module.h_docs.md)
- [`module.cpp_docs.md`](./module.cpp_docs.md)
- [`module_save.cpp_docs.md`](./module_save.cpp_docs.md)
- [`compilation_unit.h_docs.md`](./compilation_unit.h_docs.md)
- [`object.h_docs.md`](./object.h_docs.md)
- [`object.cpp_docs.md`](./object.cpp_docs.md)
- [`method.h_docs.md`](./method.h_docs.md)


## Cross-References

- **File Documentation**: `function_impl.cpp_docs.md`
- **Keyword Index**: `function_impl.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/api`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/jit/api`):

- [`compilation_unit.h_docs.md_docs.md`](./compilation_unit.h_docs.md_docs.md)
- [`object.cpp_docs.md_docs.md`](./object.cpp_docs.md_docs.md)
- [`compilation_unit.h_kw.md_docs.md`](./compilation_unit.h_kw.md_docs.md)
- [`object.h_kw.md_docs.md`](./object.h_kw.md_docs.md)
- [`module_save.cpp_kw.md_docs.md`](./module_save.cpp_kw.md_docs.md)
- [`module.h_kw.md_docs.md`](./module.h_kw.md_docs.md)
- [`method.h_kw.md_docs.md`](./method.h_kw.md_docs.md)
- [`module.h_docs.md_docs.md`](./module.h_docs.md_docs.md)
- [`method.h_docs.md_docs.md`](./method.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `function_impl.cpp_docs.md_docs.md`
- **Keyword Index**: `function_impl.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
