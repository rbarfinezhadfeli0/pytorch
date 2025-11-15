# Documentation: `docs/torch/csrc/jit/passes/inliner.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/inliner.cpp_docs.md`
- **Size**: 6,005 bytes (5.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/inliner.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/inliner.cpp`
- **Size**: 3,271 bytes (3.19 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/inliner.h>

#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch::jit {

namespace prim {
using namespace ::c10::prim;
}

GraphFunction* tryToGraphFunction(Node* n) {
  if (n->kind() == prim::CallFunction) {
    AT_ASSERT(n->input(0)->node()->kind() == prim::Constant);
    auto function_constant = n->input(0)->node();
    auto fun_type = function_constant->output()->type()->expect<FunctionType>();
    return tryToGraphFunction(*fun_type->function());
  }
  if (n->kind() == prim::CallMethod) {
    const std::string& name = n->s(attr::name);
    if (auto class_type = n->input(0)->type()->cast<ClassType>()) {
      Function& function = class_type->getMethod(name);
      return tryToGraphFunction(function);
    }
  }
  return nullptr;
}

static void inlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        if (auto graphFunction = tryToGraphFunction(cur)) {
          auto function_constant = cur->input(0)->node();
          auto fun_type =
              function_constant->output()->type()->expect<FunctionType>();

          cur->removeInput(0);
          GRAPH_UPDATE(
              "Inlining function '",
              fun_type->function()->name(),
              "' to ",
              *cur);

          std::shared_ptr<Graph> g = nullptr;
          // inline optimized graph for debugging/testing purposes.
          // we only insert fallback functions in JIT optimized graphs for
          // execution, not on the Graph that is used for serialization
          bool fallback =
              function_constant->hasAttribute(Symbol::attr("fallback"));
          if (fallback && graphFunction->get_executor().isOptimized()) {
            auto exec_plans =
                graphFunction->get_executor().getDebugState().execution_plans;
            if (!exec_plans.empty()) {
              g = exec_plans.begin()->second.graph;
              // optimized_graph() calls Inline, so we only need to explicitly
              // invoke inlining on the jit optimized graph with recursive
              // fallback function calls
              Inline(*g);
            }
          }
          if (g == nullptr) {
            g = graphFunction->optimized_graph();
          }

          GRAPH_UPDATE("Function body: ", g);
          inlineCallTo(cur, graphFunction, g.get());
        }
      } break;
      case prim::CallMethod: {
        if (auto graphFunction = tryToGraphFunction(cur)) {
          GRAPH_UPDATE("Inlining method '", cur->s(attr::name), "' to ", *cur);
          GRAPH_UPDATE("Function body: ", graphFunction->optimized_graph());
          inlineCallTo(cur, graphFunction);
        }
      } break;
      default: {
        for (auto b : cur->blocks()) {
          inlineCalls(b);
        }
      } break;
    }
  }
}

void Inline(Graph& graph) {
  GRAPH_DUMP("Before Inlining: ", &graph);
  inlineCalls(graph.block());
  GRAPH_DUMP("After Inlining: ", &graph);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `prim`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/inliner.h`
- `ATen/core/interned_strings.h`
- `torch/csrc/jit/api/function_impl.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/frontend/error_report.h`
- `torch/csrc/jit/jit_log.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/jit/passes`):

- [`inline_fork_wait.h_docs.md`](./inline_fork_wait.h_docs.md)
- [`subgraph_rewrite.cpp_docs.md`](./subgraph_rewrite.cpp_docs.md)
- [`value_refinement_utils.cpp_docs.md`](./value_refinement_utils.cpp_docs.md)
- [`create_autodiff_subgraphs.cpp_docs.md`](./create_autodiff_subgraphs.cpp_docs.md)
- [`update_differentiable_graph_requires_grad.h_docs.md`](./update_differentiable_graph_requires_grad.h_docs.md)
- [`inplace_check.h_docs.md`](./inplace_check.h_docs.md)
- [`common_subexpression_elimination.h_docs.md`](./common_subexpression_elimination.h_docs.md)
- [`dtype_analysis.cpp_docs.md`](./dtype_analysis.cpp_docs.md)
- [`canonicalize.h_docs.md`](./canonicalize.h_docs.md)
- [`add_if_then_else.h_docs.md`](./add_if_then_else.h_docs.md)


## Cross-References

- **File Documentation**: `inliner.cpp_docs.md`
- **Keyword Index**: `inliner.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/jit/passes`):

- [`peephole_dict_idioms.h_docs.md_docs.md`](./peephole_dict_idioms.h_docs.md_docs.md)
- [`remove_redundant_profiles.h_kw.md_docs.md`](./remove_redundant_profiles.h_kw.md_docs.md)
- [`loop_unrolling.cpp_kw.md_docs.md`](./loop_unrolling.cpp_kw.md_docs.md)
- [`onnx.h_kw.md_docs.md`](./onnx.h_kw.md_docs.md)
- [`guard_elimination.h_docs.md_docs.md`](./guard_elimination.h_docs.md_docs.md)
- [`frozen_conv_add_relu_fusion.cpp_docs.md_docs.md`](./frozen_conv_add_relu_fusion.cpp_docs.md_docs.md)
- [`hoist_conv_packed_params.h_kw.md_docs.md`](./hoist_conv_packed_params.h_kw.md_docs.md)
- [`lift_closures.h_kw.md_docs.md`](./lift_closures.h_kw.md_docs.md)
- [`frozen_conv_folding.h_kw.md_docs.md`](./frozen_conv_folding.h_kw.md_docs.md)
- [`frozen_graph_optimizations.h_docs.md_docs.md`](./frozen_graph_optimizations.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `inliner.cpp_docs.md_docs.md`
- **Keyword Index**: `inliner.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
