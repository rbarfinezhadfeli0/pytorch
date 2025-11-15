# Documentation: `torch/csrc/jit/runtime/interpreter/can_emit_inline.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/interpreter/can_emit_inline.h`
- **Size**: 3,954 bytes (3.86 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <memory>

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit::interpreter {
/*
This is an optimization that reduces the number of store/load/move nodes needed
by recognizing that parts of the graph are simple trees like a*x + b*y. When
this happens it is possible to work directly off of the stack by emitting the
tree in a depth-first left-to-right manner:
  load a
  load x
  mul # stack now is a*x
  load b
  load y
  mul # stack now is a*x, b*y
  add

can_emit_inline_[node] == true means that this node participates as a non-root
member of one of these trees. The code emitter will not emit this node when
it is encountered in the node. Instead the node is emitted in a depth first
traversal from where it is used in a tree.

To participate in a tree a node must have a single use (otherwise it is not
tree-like) and output a single value (for simplicity.) If our IR was functional,
these would be the only constraints. However, many nodes have side effects, so
we must ensure that emitting the nodes in depth first order from the tree's root
_does not reorder the emission of the nodes_. To ensure this, we work backward
from the root of a potential tree, visiting its inputs in reverse depth first
order, while scanning the node list backward (with the block_point node). When
these traversal line up we know it is safe to emit the tree in this way. We
ignore constant nodes, which do not have side effects.
*/
struct CanEmitInline {
  explicit CanEmitInline(Graph& graph) {
    scanBlock(graph.block());
  }
  bool canInline(Value* v) {
    return v->node()->kind() != prim::Param &&
        // without this a BailOut may float downstream past some later
        // BailOut
        // and receive a higher jf_index. Then a GUARD instruction
        // we generated for the floated BailOut will get popped up from the
        // instruction stack
        // by the later BailOut in createBailoutBlock and its jf_index
        // will become invalid.
        v->node()->kind() != prim::TensorExprGroup &&
        v->node()->kind() != prim::TensorExprDynamicGroup &&
        v->node()->kind() != prim::StaticSubgraph &&
        v->node()->kind() != prim::CudaFusionGroup &&
        v->node()->kind() != prim::FusionGroup &&
        v->node()->kind() != prim::BailOut && v->uses().size() == 1 &&
        v->node()->outputs().size() == 1;
  }

  Node* previousNonConstant(Node* n) {
    do {
      n = n->prev();
    } while (n->kind() == prim::Constant);
    return n;
  }

  Node* scanValue(Node* block_point, Value* v) {
    // this node is a candidate for inline, if our reverse scan of the
    // node list lines up with the use of v, we know it will be emitted in
    // tree order, and we can inlining. Scan continues for further nodes.
    if (v->node() == block_point && canInline(v)) {
      // since we inlined this node, we may be able to recursively inline
      // its inputs, so we continue scanning it
      block_point = scanNode(v->node());
      can_emit_inline_[v->node()] = true;
    }
    // if it does not line up, we can't inline 'v', and will just generate
    // a load/move for it. However, other inputs may still appear in tree
    // order so we continue the scan of the inputs.
    return block_point;
  }

  Node* scanNode(Node* n) {
    // don't bother to scan nodes we have already determined to be inline
    if (can_emit_inline_.count(n)) {
      return nullptr;
    }
    for (auto b : n->blocks()) {
      scanBlock(b);
    }
    Node* block_point = previousNonConstant(n);
    for (auto it = n->inputs().rbegin(), end = n->inputs().rend(); it != end;
         ++it) {
      block_point = scanValue(block_point, *it);
    }
    return block_point;
  }

  void scanBlock(Block* b) {
    scanNode(b->return_node());
    for (auto node : b->nodes().reverse()) {
      scanNode(node);
    }
  }
  std::unordered_map<Node*, bool> can_emit_inline_;
};

} // namespace torch::jit::interpreter

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `CanEmitInline`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime/interpreter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `memory`
- `torch/csrc/jit/ir/ir.h`


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

Files in the same folder (`torch/csrc/jit/runtime/interpreter`):

- [`frame.cpp_docs.md`](./frame.cpp_docs.md)
- [`code_impl.h_docs.md`](./code_impl.h_docs.md)
- [`preprocess_graph.cpp_docs.md`](./preprocess_graph.cpp_docs.md)
- [`frame.h_docs.md`](./frame.h_docs.md)
- [`preprocess_graph.h_docs.md`](./preprocess_graph.h_docs.md)


## Cross-References

- **File Documentation**: `can_emit_inline.h_docs.md`
- **Keyword Index**: `can_emit_inline.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
