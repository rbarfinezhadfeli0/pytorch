# Documentation: `docs/torch/csrc/jit/codegen/onednn/graph_rewriter.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/codegen/onednn/graph_rewriter.cpp_docs.md`
- **Size**: 7,960 bytes (7.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/codegen/onednn/graph_rewriter.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/onednn/graph_rewriter.cpp`
- **Size**: 5,301 bytes (5.18 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/codegen/onednn/graph_fuser.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch::jit::fuser::onednn {

void GraphRewriter::cleanupSubgraphs() {
  auto curNode = *block_->nodes().rbegin();
  while (curNode != *block_->nodes().rend()) {
    // Save the previous node, since we might delete `curNode` in next block
    auto prevNode = curNode->prev();
    if (llgaHelper_.isLlgaSubgraph(curNode)) {
      // Unmerge subgraph if we don't get every nodes of a partition
      // into the subgraph due to failed alias check
      llgaHelper_.unmergeIfAnyNodeIsMissing(curNode);
    }
    curNode = prevNode;
  }
  for (Node* n : block_->nodes()) {
    for (Block* b : n->blocks()) {
      GraphRewriter(b, graph_, aliasDb_).cleanupSubgraphs();
    }
  }
}

void GraphRewriter::buildupSubgraphs() {
  // We need to run the rewriter multiple times in order to get all merge
  // opportunities. This is because moveBeforeTopologicalValid may reorder
  // nodes to be AFTER the current iteration point. In order to properly
  // consider those nodes for merging, we need run the pass until no changes
  // have been made.
  //
  // Example:
  //   c = f(a, b)
  //   d = f(c)
  //   e = f(d)  <- iter is here, moving upward
  // After c.moveBeforeTopologicallyValid(e), we have:
  //   c = f(a, b)
  //   e = f(d)  <- iter still here
  //   d = f(c)  <- this was node moved on the other side.
  // see [workblocks]
  auto workblocks = buildWorkBlocks();
  for (auto& workblock : workblocks) {
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      auto workblock_end = workblock.end()->reverseIterator();
      auto workblock_begin = workblock.begin()->reverseIterator();
      for (auto it = workblock_end; it != workblock_begin;) {
        bool changed = false;
        std::tie(it, changed) = scanNode(*it, workblock_begin);
        any_changed |= changed;
      }
    }
  }

  // Construct Subgraphs Recursively
  for (Node* n : block_->nodes()) {
    for (auto subBlock : n->blocks()) {
      GraphRewriter(subBlock, graph_, aliasDb_).buildupSubgraphs();
    }
  }
}

std::vector<WorkBlock> GraphRewriter::buildWorkBlocks() {
  // [workblocks]
  // the IR has many nodes which can never be reordered around, such as a
  // prim::Bailout. if a node N is surrounded by two nodes which cannot be
  // reordered, A and B, then a fusion group that is created from N
  // can only contain nodes from (A, B) The nodes from A to B represent one
  // work block for the subgraph rewriter to work on. By creating these up
  // front, we avoid retraversing the whole graph block any time scanNode
  // returns
  Node* end_bound_node = block_->return_node();
  Node* curr = end_bound_node->prev();
  std::vector<WorkBlock> worklist;
  while (curr != block_->param_node()) {
    // cannot reorder around side effectful nodes
    if (curr->hasSideEffects()) {
      worklist.emplace_back(curr, end_bound_node);
      end_bound_node = curr;
    }
    curr = curr->prev();
  }
  worklist.emplace_back(curr, end_bound_node);
  return worklist;
}

std::pair<graph_node_list::iterator, bool> GraphRewriter::scanNode(
    Node* consumer,
    graph_node_list::iterator workblock_begin) {
  GRAPH_DEBUG("Scanning ", consumer->kind().toQualString());
  if (llgaHelper_.shouldConsiderForMerge(consumer)) {
    if (!llgaHelper_.isLlgaSubgraph(consumer)) {
      consumer = llgaHelper_.createSingletonSubgraph(consumer, aliasDb_);
    }
    // Iterate through the workblock to merge nodes of the
    // same partition determined by LLGA graph helper.
    // Nodes like B and C do not share a common input but belong to a
    // same partition, and thus we cannot only scan the input nodes
    // to find merging opportunities. Instead, we have to scan through
    // the whole workblock, which might lead to O^2 accesses in worst case
    //              A
    //      + - - / - \ - - +
    //      |    B     C    |
    //      |    |     |    |
    //      |    D     E    |
    //      + - - \ - / - - +
    //              F
    auto prev = ++consumer->reverseIterator();
    for (auto it = prev; it != workblock_begin; it++) {
      if (auto group = tryMerge(consumer, *it)) {
        // we successfully merged, so the new group's `inputs` may have
        // changed. So rescan the new group for more merging opportunities.
        return std::make_pair(group.value()->reverseIterator(), true);
      }
    }
  }
  return std::make_pair(++consumer->reverseIterator(), false);
}

// Try to merge `producer` into `consumer`. If successful, this destroys
// `producer` and returns the `consumer` group.
std::optional<Node*> GraphRewriter::tryMerge(Node* consumer, Node* producer) {
  AT_ASSERT(llgaHelper_.isLlgaSubgraph(consumer));
  bool canMerge = llgaHelper_.shouldMerge(producer, consumer) &&
      aliasDb_.moveBeforeTopologicallyValid(producer, consumer);
  if (!canMerge) {
    return std::nullopt;
  }
  llgaHelper_.mergeNodeIntoSubgraph(producer, consumer, aliasDb_);
  return consumer;
}

} // namespace torch::jit::fuser::onednn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Subgraphs`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/codegen/onednn/graph_fuser.h`
- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/common_subexpression_elimination.h`
- `torch/csrc/jit/passes/dead_code_elimination.h`
- `torch/csrc/jit/passes/utils/subgraph_utils.h`


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

Files in the same folder (`torch/csrc/jit/codegen/onednn`):

- [`guard_shape.cpp_docs.md`](./guard_shape.cpp_docs.md)
- [`prepare_binary.h_docs.md`](./prepare_binary.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`graph_fuser.h_docs.md`](./graph_fuser.h_docs.md)
- [`kernel.h_docs.md`](./kernel.h_docs.md)
- [`decompose_silu.cpp_docs.md`](./decompose_silu.cpp_docs.md)
- [`prepare_binary.cpp_docs.md`](./prepare_binary.cpp_docs.md)
- [`graph_helper.cpp_docs.md`](./graph_helper.cpp_docs.md)
- [`register_interface.cpp_docs.md`](./register_interface.cpp_docs.md)


## Cross-References

- **File Documentation**: `graph_rewriter.cpp_docs.md`
- **Keyword Index**: `graph_rewriter.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/codegen/onednn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/codegen/onednn`):

- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)
- [`decompose_silu.cpp_kw.md_docs.md`](./decompose_silu.cpp_kw.md_docs.md)
- [`defer_size_check.h_kw.md_docs.md`](./defer_size_check.h_kw.md_docs.md)
- [`graph_fuser.h_kw.md_docs.md`](./graph_fuser.h_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`graph_fuser.h_docs.md_docs.md`](./graph_fuser.h_docs.md_docs.md)
- [`interface.h_docs.md_docs.md`](./interface.h_docs.md_docs.md)
- [`layout_propagation.h_kw.md_docs.md`](./layout_propagation.h_kw.md_docs.md)
- [`graph_helper.cpp_kw.md_docs.md`](./graph_helper.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `graph_rewriter.cpp_docs.md_docs.md`
- **Keyword Index**: `graph_rewriter.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
