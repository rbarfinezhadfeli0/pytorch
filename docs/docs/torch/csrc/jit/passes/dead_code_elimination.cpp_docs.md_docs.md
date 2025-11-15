# Documentation: `docs/torch/csrc/jit/passes/dead_code_elimination.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/dead_code_elimination.cpp_docs.md`
- **Size**: 18,708 bytes (18.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/dead_code_elimination.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/dead_code_elimination.cpp`
- **Size**: 15,893 bytes (15.52 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>

#include <unordered_map>

namespace torch::jit {

namespace prim {
using namespace ::c10::prim;
}

class DeadCodeEliminator {
 public:
  explicit DeadCodeEliminator(
      std::shared_ptr<Graph> graph,
      DCESideEffectPolicy sideEffectPolicy)
      : sideEffectPolicy_(sideEffectPolicy),
        graph_(std::move(graph)),
        useAliasDb_(true) {}
  DeadCodeEliminator(DCESideEffectPolicy sideEffectPolicy)
      : sideEffectPolicy_(sideEffectPolicy) {}

  // The algorithm is an inverse mark-and-sweep. Starting from the return node,
  // we mark "live" nodes that are necessary for the output. Nodes that have
  // side effects are also marked.
  void run(Block* block, bool recurse) {
    // clean up unused fork inputs before starting the main algorithm
    eliminateDeadForkInputs(block, recurse);

    // Initialize by marking the return node and all its consumed values as live
    mark(block->return_node());

    mark(block);

    deleteCallback_(getLiveValues());

    sweep(block, recurse);
  }

  void setDeleteCallback(
      std::function<void(const std::unordered_set<const Value*>&)>
          deleteCallback) {
    deleteCallback_ = std::move(deleteCallback);
  }

 private:
  void eliminateDeadForkInputs(Block* block, bool recurse) {
    for (Node* node : block->nodes()) {
      if (recurse) {
        for (Block* sb : node->blocks()) {
          eliminateDeadForkInputs(sb, recurse);
        }
      }
      if (node->kind() != prim::fork) {
        continue;
      }
      Graph& g = *node->g(attr::Subgraph);
      // WARNING: Do not use a ranged loop. The loop bounds are changed by the
      // loop body.
      for (size_t i = 0; i < g.inputs().size(); ++i) {
        if (!g.inputs().at(i)->hasUses()) {
          GRAPH_UPDATE(
              "Dead ",
              i,
              "-th input ",
              node->inputs().at(i)->debugName(),
              "(",
              g.inputs().at(i)->debugName(),
              " in a subgraph) will be removed");
          g.eraseInput(i);
          node->removeInput(i);
        }
      }
    }
  }

  // Special handling for block return nodes. Unlike other nodes, the block
  // return node doesn't really "use" its inputs. Consider:
  //
  // %a0 = aten::foo()
  // %b = aten::foo()
  // %a2, %b2 = prim::If(%cond) {
  //   block0() {
  //     %a1 = aten::foo(%.0)
  //     %b1 = aten::foo(%b)
  //   } -> (%a1, %b1)
  // }
  // return (%a2)
  //
  // We want to be able to DCE all the %b stuff. So when processing block
  // returns, we only mark producers for values that "live" (i.e. used outside
  // the block).
  //
  // Returns true iff this marked something we haven't marked before.
  bool markReturnNode(Node* node) {
    if (marked_.count(node)) {
      return false;
    }

    AT_ASSERT(node->owningBlock()->return_node() == node);
    auto outerNode = node->owningBlock()->owningNode();
    if (outerNode == nullptr || outerNode->kind() == prim::Reverse) {
      // If there's no outer node, we're looking at the graph's top-level
      // return block. We consider all graph outputs to be "used", so just mark
      // this node normally.
      return mark(node);
    }

    // Collect all inputs that are actually live
    if (outerNode->kind() == prim::Loop ||
        outerNode->kind() == c10::onnx::Loop) {
      // Special handling to deal with loop carried dependencies.
      auto loop = LoopView(outerNode);
      for (const auto i : c10::irange(loop.carriedOutputs().size())) {
        if (outerNode->kind() == c10::onnx::Loop) {
          // Special handling for onnx loop.
          // The number of body carried inputs and outputs are different.
          // They cannot be mapped to each other easily by the same index.
          insertLiveValue(loop.bodyCarriedOutputs().at(i));
          continue;
        }
        auto innerInput = loop.bodyCarriedInputs().at(i);
        auto innerOutput = loop.bodyCarriedOutputs().at(i);
        auto outerOutput = loop.carriedOutputs().at(i);
        if (liveValuesContains(outerOutput) || innerInput->hasUses()) {
          insertLiveValue(innerOutput);
        }
      }

      // Also mark the loop next condition as live, since it will be used inside
      // the loop body.
      insertLiveValue(loop.nextCond());
    } else {
      AT_ASSERT(outerNode->outputs().size() == node->inputs().size());
      for (const auto i : c10::irange(outerNode->outputs().size())) {
        auto innerOutput = node->inputs()[i];
        auto outerOutput = outerNode->outputs()[i];
        if (liveValuesContains(outerOutput)) {
          insertLiveValue(innerOutput);
        }
      }
    }

    marked_.insert(node);
    return true;
  }

  // Loops are special, because we need to run them to convergence.
  // Consider the following loop:
  //   for i in range(3):
  //     tot += a[0][0]
  //     b = a[0]
  //     b[0] += 1
  //   print(tot)
  //
  // If we only process the loop block once, we will conclude that `b[0]` and
  // `b` are dead, even though `b[0] += 1` mutates a live memory location (since
  // `b[0]` is an alias of `a`). i.e. `a` is used to compute `tot` in the next
  // iteration
  //
  // We need to mark the loop again with the information that `a` is live, and
  // repeat until we're not marking new stuff anymore.
  //
  // Returns true iff this marked something we haven't marked before.
  bool markLoop(Node* node) {
    TORCH_INTERNAL_ASSERT(node->kind() == prim::Loop);
    // Did a single iteration over the loop block mark anything new?
    // If this is false, we've converged.
    bool marked = false;
    // Did we ever mark anything new?
    bool anyMarked = false;
    do {
      marked = mark(node->blocks().at(0));
      anyMarked |= marked;
    } while (marked);
    return anyMarked;
  }

  // Returns true iff this marked something we haven't marked before.
  bool mark(Block* block) {
    bool anyMarked = false;
    // Mark all nodes with side effects.
    for (auto node : block->nodes()) {
      if (sideEffectPolicy_ ==
              DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS &&
          hasSideEffects(node)) {
        anyMarked |= mark(node);
      }
    }

    // Initialize by marking the return node
    anyMarked |= markReturnNode(block->return_node());

    for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); ++it) {
      auto node = *it;
      if (node->kind() == prim::Loop) {
        // Special casing for loops, see comment in markLoop.
        anyMarked |= markLoop(node);
      } else {
        // Other nodes with sub-blocks get marked normally.
        for (auto subBlock : node->blocks()) {
          anyMarked |= mark(subBlock);
        }
      }
      anyMarked |= markIfLive(node);
    }
    return anyMarked;
  }

  // If we output or write to a live memory location, mark this node
  // Returns true iff this marked something we haven't marked before.
  bool markIfLive(Node* node) {
    for (const auto output : node->outputs()) {
      if (liveValuesContains(output)) {
        return mark(node);
      }
    }

    if (useAliasDb_) {
      if (getOrCreateAliasDb()->writesToAlias(
              node, getLiveValuesAndMemoryLocations())) {
        return mark(node);
      }
    }

    return false;
  }

  // Mark this node as live and add this node's inputs and aliases to the live
  // value sets.
  // Returns true iff this marked something we haven't marked before.
  bool mark(Node* node) {
    if (marked_.count(node)) {
      return false;
    }

    marked_.insert(node);

    // Mark all nodes in this node's blockchain (since owning nodes are
    // considered live if they contain a live node)
    auto curNode = node;
    while (curNode) {
      if (!curNode->owningBlock()) {
        break;
      }

      mark(curNode);
      curNode = curNode->owningBlock()->owningNode();
    }

    for (const auto input : node->inputs()) {
      if (liveValuesContains(input)) {
        continue;
      }
      insertLiveValue(input);
    }
    return true;
  }

  // Delete all unmarked nodes.
  void sweep(Block* block, bool recurse) {
    auto nodes = block->nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      // note these occur before the recursion because we want to uncover
      // dead code in the blocks used to calculate the output
      removeDeadBlockOutputs(node);
      removeDeadLoopOutputs(node);
      if (recurse) {
        for (Block* block : node->blocks()) {
          sweep(block, true);
        }
      }
      // NB: Checking hasUses() is required. AD graphs are not perfectly
      // valid, as a node in grad_desc.f might be used in reverse_block.
      // Reverse_block is inlined in grad_desc.f before it's separated
      // to grad_desc.df.
      if (!(marked_.count(node) || node->hasUses())) {
        GRAPH_UPDATE(
            "Node ",
            it->kind().toQualString(),
            " which outputs ",
            (!node->outputs().empty() ? node->outputs().at(0)->debugName()
                                      : "n/a"),
            " will be removed");
        it.destroyCurrent();
      }
    }
  }

  bool hasUntrackedMutation(Node* node) {
    if (!useAliasDb_) {
      // If we don't have alias information, all mutable ops have unknown
      // effects and can't be considered for elimination.

      if (node->kind() == prim::SetAttr) {
        // SetAttr is a special case: it doesn't have a schema, but does
        // have untracked mutations
        return true;
      }

      // onnx export calls EliminateDeadCode but sometimes passes invalid
      // aten operators. So we call maybeSchema so we handle the cases when
      // there is no valid schema for a node
      auto schema = node->maybeSchema();
      return schema && schema->is_mutable();
    } else {
      return getOrCreateAliasDb()->writesToWildcard(node);
    }
  }

  bool hasSideEffects(Node* node) {
    auto it = memo_.find(node);
    if (it != memo_.end())
      return it->second;
    bool has_side_effects = node->hasSideEffects() ||
        std::any_of(node->blocks().begin(),
                    node->blocks().end(),
                    [&](Block* b) {
                      return std::any_of(
                          b->nodes().begin(), b->nodes().end(), [&](Node* n) {
                            return hasSideEffects(n);
                          });
                    }) ||
        hasUntrackedMutation(node);

    memo_.emplace(node, has_side_effects);
    return has_side_effects;
  }

  void removeDeadBlockOutputs(Node* node) {
    if (node->kind() != prim::If && node->kind() != prim::GradOf) {
      return;
    }

    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses()) {
        GRAPH_UPDATE(
            "Dead ",
            i,
            "-th output ",
            node->outputs().at(i)->debugName(),
            " of node ",
            node->kind().toQualString(),
            " will be removed");
        node->eraseOutput(i);
        for (Block* b : node->blocks()) {
          GRAPH_UPDATE(
              "\tCorresponding block output ",
              b->outputs().at(i)->debugName(),
              " will be removed");
          b->eraseOutput(i);
        }
      }
    }
  }

  void removeDeadLoopOutputs(Node* node) {
    if (node->kind() != prim::Loop)
      return;
    auto loop_body = node->blocks().at(0);
    auto loop_input_offset = 2; // offset of loop carried deps in input list
    auto loop_body_offset =
        1; // offset to the loop carried dependencies in block inputs/outputs

    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses() &&
          !loop_body->inputs().at(loop_body_offset + i)->hasUses()) {
        logDeadLoopOutputs(node, i, loop_input_offset, loop_body_offset);
        node->eraseOutput(i);
        node->removeInput(loop_input_offset + i);
        loop_body->eraseInput(loop_body_offset + i);
        loop_body->eraseOutput(loop_body_offset + i);
      }
    }
  }

  void logDeadLoopOutputs(
      Node* node,
      size_t i,
      size_t loop_input_offset,
      size_t loop_body_offset) {
    auto loop_body = node->blocks().at(0);
    GRAPH_UPDATE(
        "Dead ",
        loop_input_offset + i,
        "-th input ",
        node->inputs().at(i)->debugName(),
        " will be removed");
    GRAPH_UPDATE(
        "Dead ",
        i,
        "-th output ",
        node->outputs().at(i)->debugName(),
        " will be removed");
    GRAPH_UPDATE(
        "\tDead block input ",
        loop_body->inputs().at(loop_body_offset + i)->debugName(),
        "at offset ",
        loop_body_offset + i,
        " will be removed");
    GRAPH_UPDATE(
        "\tDead block output ",
        loop_body->outputs().at(loop_body_offset + i)->debugName(),
        "at offset ",
        loop_body_offset + i,
        " will be removed");
  }

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  ValueAndMemoryLocationSet& getLiveValuesAndMemoryLocations() {
    if (!liveValuesAndMemoryLocations_) {
      liveValuesAndMemoryLocations_ =
          std::make_unique<ValueAndMemoryLocationSet>(
              getOrCreateAliasDb()->getValueAndMemoryLocationSet());
    }
    return *liveValuesAndMemoryLocations_;
  }

  ValueSet& getLiveValuesSet() {
    if (!liveValuesSet_) {
      liveValuesSet_ = std::make_unique<ValueSet>();
    }
    return *liveValuesSet_;
  }

  ValueSet& getLiveValues() {
    if (useAliasDb_) {
      return getLiveValuesAndMemoryLocations().getValueSet();
    } else {
      return getLiveValuesSet();
    }
  }

  void insertLiveValue(Value* v) {
    if (useAliasDb_) {
      getLiveValuesAndMemoryLocations().insert(v);
    } else {
      getLiveValuesSet().insert(v);
    }
  }

  bool liveValuesContains(Value* v) {
    if (useAliasDb_) {
      return getLiveValuesAndMemoryLocations().getValueSet().count(v);
    } else {
      return getLiveValuesSet().count(v);
    }
  }

  DCESideEffectPolicy sideEffectPolicy_;

  std::shared_ptr<Graph> graph_;
  bool useAliasDb_ = false;
  // lazily initialized
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::unordered_map<Node*, bool> memo_;
  std::unordered_set<Node*> marked_;

  // we should have at most 1 of these as a non-nullptr; they are lazily
  // initialized. liveValuesAndMemoryLocations_ is used if we are using AliasDb
  //   (in order to store aliasing info),
  // otherwise liveValuesSet_ is used.
  std::unique_ptr<ValueAndMemoryLocationSet> liveValuesAndMemoryLocations_ =
      nullptr;
  std::unique_ptr<ValueSet> liveValuesSet_ = nullptr;

  std::function<void(const std::unordered_set<const Value*>&)> deleteCallback_ =
      [](const std::unordered_set<const Value*>&) {};
};

void EliminateDeadCode(
    const std::shared_ptr<Graph>& graph,
    DCESideEffectPolicy sideEffectPolicy) {
  DeadCodeEliminator(graph, sideEffectPolicy)
      .run(graph->block(), /*recurse=*/true);
  GRAPH_DUMP("After EliminateDeadCode: ", graph);
}

void EliminateDeadCode(
    Block* block,
    bool recurse,
    DCESideEffectPolicy sideEffectPolicy) {
  DeadCodeEliminator(sideEffectPolicy).run(block, recurse);
}

void EliminateDeadCode(
    Block* block,
    std::function<void(const std::unordered_set<const Value*>&)> cb,
    DCESideEffectPolicy sideEffectPolicy) {
  DeadCodeEliminator eliminator(sideEffectPolicy);
  eliminator.setDeleteCallback(std::move(cb));
  eliminator.run(block, /*recurse=*/true);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 39 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `prim`

**Classes/Structs**: `DeadCodeEliminator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/dead_code_elimination.h`
- `c10/util/irange.h`
- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/ir/ir_views.h`
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

- **File Documentation**: `dead_code_elimination.cpp_docs.md`
- **Keyword Index**: `dead_code_elimination.cpp_kw.md`
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

- **File Documentation**: `dead_code_elimination.cpp_docs.md_docs.md`
- **Keyword Index**: `dead_code_elimination.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
