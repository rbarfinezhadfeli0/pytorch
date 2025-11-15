# Documentation: `docs/torch/csrc/jit/passes/common_subexpression_elimination.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/common_subexpression_elimination.cpp_docs.md`
- **Size**: 6,801 bytes (6.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/common_subexpression_elimination.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/common_subexpression_elimination.cpp`
- **Size**: 3,914 bytes (3.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>

#include <unordered_map>

namespace torch::jit {
namespace {

struct CommonSubexpressionEliminator {
  CommonSubexpressionEliminator(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run(std::function<Node*(Node*)> parent_lookup_fn) {
    return run(graph_->block(), std::move(parent_lookup_fn));
  }

  // The function implements common subexpression elimination.
  // Since the nodes are visited in topological order, one pass is enough.
  // returns true if CSE made changes to a graph
  bool run(Block* block, std::function<Node*(Node*)> parent_lookup_fn) {
    std::unordered_set<Node*, HashNode, EqualNode> subexprs;
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto node = *it;

      if (node->kind() == prim::profile) {
        GRAPH_DEBUG(
            "Profiled nodes shouldn't be CSE'ed there's a separate pass that does dedup and merging:\n",
            *node);
        continue;
      }

      if (node->hasSideEffects()) {
        GRAPH_DEBUG("Node was skipped due to side effects:\n", *node);
        continue;
      }
      if (node->isNondeterministic()) {
        GRAPH_DEBUG("Node was skipped due to its non determinism:\n", *node);
        continue;
      }

      if (!node->blocks().empty()) {
        // Traverse sub-blocks.
        for (auto block : node->blocks()) {
          changed |= run(block, [&](Node* n) {
            auto existing = subexprs.find(n);
            if (existing != subexprs.end()) {
              return *existing;
            }

            return parent_lookup_fn(n);
          });
        }

        continue;
      }

      if (getOrCreateAliasDb().hasWriters(node)) {
        GRAPH_DEBUG("Node was skipped due to alias analysis result:\n", *node);
        // Do NOT have enough information to do CSE on these nodes.
        continue;
      }

      // Check for CSE opportunities in the parent block.
      auto parent_lookup = parent_lookup_fn(node);
      auto g_out = node->owningGraph()->outputs();
      if (parent_lookup != nullptr) {
        if (!getOrCreateAliasDb().safeToChangeAliasingRelationship(
                node->outputs(), parent_lookup->outputs())) {
          continue;
        }

        GRAPH_UPDATE("Replacing\n", *node, "with\n", *parent_lookup);
        changed = true;
        node->replaceAllUsesWith(parent_lookup);
        it.destroyCurrent();
        continue;
      }

      // Check whether the same subexpression already exists.
      auto subit = subexprs.insert(node);
      if (!subit.second) {
        // Subexpression exists, replace the uses of node, and destroy it.
        auto existing = *subit.first;

        // don't introduce new aliasing among graph outputs
        if (getOrCreateAliasDb().mayContainAlias(
                node->outputs(), node->owningGraph()->outputs()) &&
            getOrCreateAliasDb().mayContainAlias(existing->outputs(), g_out)) {
          continue;
        }

        GRAPH_UPDATE("Replacing\n", *node, "with\n", *existing);
        changed = true;
        node->replaceAllUsesWith(existing);
        // Destroy the node.
        it.destroyCurrent();
      }
    }

    return changed;
  }

  AliasDb& getOrCreateAliasDb() {
    if (!alias_db_) {
      alias_db_ = std::make_unique<AliasDb>(graph_);
    }

    return *alias_db_;
  }

 private:
  std::unique_ptr<AliasDb> alias_db_;
  std::shared_ptr<Graph> graph_;
};

} // namespace

bool EliminateCommonSubexpression(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before CSE", graph);
  CommonSubexpressionEliminator cse(graph);
  return cse.run([](Node*) { return nullptr; });
}
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `bool`

**Classes/Structs**: `CommonSubexpressionEliminator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/common_subexpression_elimination.h`
- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/node_hashing.h`
- `torch/csrc/jit/jit_log.h`
- `unordered_map`


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

- **File Documentation**: `common_subexpression_elimination.cpp_docs.md`
- **Keyword Index**: `common_subexpression_elimination.cpp_kw.md`
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

- **File Documentation**: `common_subexpression_elimination.cpp_docs.md_docs.md`
- **Keyword Index**: `common_subexpression_elimination.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
