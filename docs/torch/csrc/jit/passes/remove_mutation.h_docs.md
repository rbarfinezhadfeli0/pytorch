# Documentation: `torch/csrc/jit/passes/remove_mutation.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/remove_mutation.h`
- **Size**: 2,637 bytes (2.58 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>

#include <utility>

namespace torch::jit {

struct TORCH_API MutationRemover {
  MutationRemover(
      std::shared_ptr<Graph> graph,
      std::optional<std::function<bool(Node*)>> mutation_filter = std::nullopt)
      : mutation_filter_(std::move(mutation_filter)),
        aliasDb_(nullptr),
        graph_(std::move(graph)) {}

  // return true if graph is modified
  bool removeListMutation();

  // return true if graph is modified
  bool removeTensorMutation();

  bool isSpecialMappedOp(Node* n) {
    return n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)") ||
        n->matches(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)") ||
        n->matches(
            "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)");
  }

  bool inplaceOpVariant(Node* n);

  static bool hasSideEffectOrAlias(Value* v, AliasDb* aliasDb);

 private:
  Node* createSpecialMappedOp(Node* n);
  bool listMutationFollowingListConstruct(Node* n);
  bool tryMakeCreationAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  bool tryMakeUnaliasedIfOutputAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  // return true if graph is modified
  bool RemoveListMutation(Block* block);
  // return true if graph is modified
  bool RemoveTensorMutation(Block* block);

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  std::optional<std::function<bool(Node*)>> mutation_filter_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

// Removes list mutation with functional equivalents
// return true if graph is modified
TORCH_API bool RemoveListMutation(const std::shared_ptr<Graph>& graph);

// Replaces in-place aten ops with their functional equivalents
// when it can be proven that this does not change graph semantics
// if `mutation_filter` is present, the pass will only attempt to
// remove mutation on nodes which return true for the filter
// return true if graph is modified
TORCH_API bool RemoveTensorMutation(
    const std::shared_ptr<Graph>& graph,
    std::optional<std::function<bool(Node*)>> mutation_filter = std::nullopt);

// Replaces in-place aten activation ops with their functional equivalence
TORCH_API bool InplaceToFunctionalActivation(
    const std::shared_ptr<Graph>& graph);

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `torch/csrc/Export.h`
- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/ir/ir.h`
- `utility`


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

- **File Documentation**: `remove_mutation.h_docs.md`
- **Keyword Index**: `remove_mutation.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
