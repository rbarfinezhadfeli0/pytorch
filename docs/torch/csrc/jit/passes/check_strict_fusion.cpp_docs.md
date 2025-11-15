# Documentation: `torch/csrc/jit/passes/check_strict_fusion.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/check_strict_fusion.cpp`
- **Size**: 3,820 bytes (3.73 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp

#include <torch/csrc/jit/passes/check_strict_fusion.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch::jit {

namespace {

bool isStrictFusion(Value* value) {
  const auto class_name = getModuleName(value);
  return class_name.has_value() &&
      (*class_name == "__torch__.torch.jit.strict_fusion");
}

} // namespace

static bool fusionGuardCheck(Symbol k) {
  return k == Symbol::prim("TensorExprDynamicGuard") || k == prim::TypeCheck ||
      k == prim::CudaFusionGuard || k == prim::RequiresGradCheck;
}

static std::unordered_set<Node*> collectValuesUsedInGuard(
    Node* guarding_if,
    Node* enter_node) {
  // DFS to collect
  std::unordered_set<Node*> visited_nodes;
  std::vector<Node*> queue = {guarding_if};

  while (!queue.empty()) {
    Node* curr = queue[queue.size() - 1];
    queue.pop_back();
    visited_nodes.insert(curr);
    // these nodes directly test Tensor inputs, and are not part of additional
    // guards inserted
    if (fusionGuardCheck(curr->kind())) {
      continue;
    }
    for (Value* v : curr->inputs()) {
      Node* inp_node = v->node();
      if (inp_node->isBefore(enter_node) ||
          inp_node->owningBlock() != enter_node->owningBlock()) {
        continue;
      }
      if (visited_nodes.count(inp_node)) {
        continue;
      }
      queue.push_back(inp_node);
    }
  }
  return visited_nodes;
}

static void checkForUnfusedOps(Node* enter_node) {
  std::vector<Node*> unsupported_nodes;
  std::vector<Node*> guarding_ifs; // if multiple, we will throw
  for (Node* node = enter_node->next(); node->kind() != prim::Exit;
       node = node->next()) {
    if (node->kind() == prim::If &&
        fusionGuardCheck(node->input()->node()->kind())) {
      guarding_ifs.push_back(node);
      continue;
    }
    unsupported_nodes.push_back(node);
  }

  if (guarding_ifs.size() > 1) {
    std::stringstream ss;
    ss << "Found multiple fusions: \n";
    for (Node* n : guarding_ifs) {
      ss << *n << "\n";
    }
    throw(ErrorReport(enter_node->input()->node()->sourceRange()) << ss.str());
  }

  // autodiff/nnc both insert a number of guards, see
  // `CudaFusionViewGuard Example Graph`
  // to check for unfused nodes, look at node's whose outputs
  // are not depended on by the fusion guard
  // restrict search for all values after the first
  // node in the prim::Enter block

  std::unordered_set<Node*> guarding_check_nodes;
  if (guarding_ifs.size() == 1) {
    guarding_check_nodes =
        collectValuesUsedInGuard(guarding_ifs[0], enter_node);
  }
  std::vector<Node*> unfused_nodes_not_used_in_guard;
  for (Node* unfused : unsupported_nodes) {
    if (!guarding_check_nodes.count(unfused)) {
      unfused_nodes_not_used_in_guard.push_back(unfused);
    }
  }
  if (!unfused_nodes_not_used_in_guard.empty()) {
    std::stringstream ss;
    ss << "Found unfused operators: \n";
    for (Node* unfused : unfused_nodes_not_used_in_guard) {
      ss << "\t";
      if (unfused->maybeSchema()) {
        ss << unfused->schema();
      } else {
        unfused->kind().toDisplayString();
      }
      ss << "\n";
    }
    throw(ErrorReport(enter_node->input()->node()->sourceRange()) << ss.str());
  }
}

void CheckStrictFusion(std::shared_ptr<Graph>& graph) {
  DepthFirstGraphNodeIterator it(graph);
  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == prim::Enter && isStrictFusion(n->input())) {
      checkForUnfusedOps(n);
    }
  }

  // TODO: remove context manager after checks
  // TODO: improve control flow not taken, right now always errors
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `static`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/check_strict_fusion.h`
- `c10/util/Exception.h`
- `torch/csrc/jit/frontend/error_report.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/quantization/helper.h`
- `torch/csrc/jit/runtime/graph_iterator.h`


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

- **File Documentation**: `check_strict_fusion.cpp_docs.md`
- **Keyword Index**: `check_strict_fusion.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
