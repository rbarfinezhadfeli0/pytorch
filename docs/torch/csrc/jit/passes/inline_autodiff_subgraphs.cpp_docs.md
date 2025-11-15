# Documentation: `torch/csrc/jit/passes/inline_autodiff_subgraphs.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/inline_autodiff_subgraphs.cpp`
- **Size**: 2,647 bytes (2.58 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch::jit {

// aten and prim nodes (except FusionGroup) are guaranteed to work
// with Autograd, other nodes (e.g. user-defined nodes) are not necessarily
// Autograd-aware
bool canRunWithAutograd(Node* node) {
  auto kind = node->kind();
  for (Block* block : node->blocks()) {
    if (!std::all_of(
            block->nodes().begin(), block->nodes().end(), canRunWithAutograd)) {
      return false;
    }
  }
  return kind != prim::FusionGroup && kind != prim::CudaFusionGroup &&
      kind != prim::TypeCheck && kind != prim::TensorExprGroup &&
      kind != prim::CudaFusionGuard && kind != prim::oneDNNFusionGroup &&
      kind != prim::oneDNNFusionGuard && (kind.is_aten() || kind.is_prim());
}

namespace {

void InlineAutodiffSubgraphs(Block* block, size_t threshold);

size_t blockSize(Block* block) {
  size_t num = 0;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      num += blockSize(b);
    }
    num++;
  }
  return num;
}

graph_node_list::iterator scanNode(Node* node, size_t threshold) {
  auto next_node = ++node->iterator();

  for (Block* block : node->blocks()) {
    InlineAutodiffSubgraphs(block, threshold);
  }

  if (node->kind() != prim::DifferentiableGraph) {
    return next_node;
  }

  auto subgraph = node->g(attr::Subgraph);
  size_t subgraph_size = blockSize(subgraph->block());
  if (subgraph_size >= threshold) {
    return next_node;
  }

  if (!std::all_of(
          subgraph->nodes().begin(),
          subgraph->nodes().end(),
          canRunWithAutograd)) {
    return next_node;
  }

  // now that we inline the graph, we are no longer detaching input tensors,
  // so the profiles will have outdated requires_grad=False.
  // conservatively update them to maybe requiring grad, bc we might create
  // autodiff graphs when the tensors maybe require grad
  UpdateDifferentiableGraphRequiresGrad(subgraph, std::nullopt);
  SubgraphUtils::unmergeSubgraph(node);
  return next_node;
}

void InlineAutodiffSubgraphs(Block* block, size_t threshold) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    it = scanNode(*it, threshold);
  }
}

} // anonymous namespace

void InlineAutodiffSubgraphs(std::shared_ptr<Graph>& graph, size_t threshold) {
  InlineAutodiffSubgraphs(graph->block(), threshold);
  EliminateDeadCode(graph);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/inline_autodiff_subgraphs.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/passes/dead_code_elimination.h`
- `torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h`
- `torch/csrc/jit/passes/utils/subgraph_utils.h`


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

- **File Documentation**: `inline_autodiff_subgraphs.cpp_docs.md`
- **Keyword Index**: `inline_autodiff_subgraphs.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
