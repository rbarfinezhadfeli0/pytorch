# Documentation: `torch/csrc/jit/runtime/graph_iterator.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/graph_iterator.h`
- **Size**: 4,938 bytes (4.82 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// This class facilitates depth-first iteration over all nodes in a graph.
class DepthFirstGraphNodeIterator {
  Node* current_;

 public:
  // Constructor.
  explicit DepthFirstGraphNodeIterator(std::shared_ptr<Graph>& graph)
      : current_(*(graph->block()->nodes().begin())) {}

  // Moves up and to the next node (may move up recursively).
  void move_up() {
    if (current_ == nullptr) {
      return;
    }
    // Basically we start from the child block (which is current_)
    // and we try to find the block that owns it. Now we need to check
    // if that block is the graph root block, or if it is an If/Loop/etc
    // block.
    //
    // If it's the graph root block we can stop because there is no "up"
    // but if it is a node (e.g. If/Loop/etc) we need to apply logic
    // based on where we are coming from to move to the next block.
    // This might mean that we need to traverse up again (e.g. if we've
    // reached the end of the else clause in an if block we need to go)
    // up to the parent block that contains the if.
    //
    // Similarly if we've reached the end of the parent block containing
    // the else clause we might need to go up again so this is a recursive
    // function.
    //
    //              BlockNode (if/loop/with)
    //                       |
    //            [Block1]  ... [Block2]
    //                |
    //   [ Node1, Node2, Node3, FromNode]
    //
    auto parent_block = current_->owningBlock();
    TORCH_INTERNAL_ASSERT(parent_block, "Every node must be owned by a block");

    // Get the node that owns the parent block. This node has to be an if,
    // loop, or with.
    auto parent_node = parent_block->owningNode();
    if (parent_node == nullptr) {
      // If there's no node that owns this current block then we're at the
      // top of the graph and since we're trying to move up we have reached
      // the end of the traversal.
      current_ = nullptr;
      return;
    }

    // Check the type of node this root is.
    if (parent_node->kind() == prim::If) {
      // Need to check if we came from the `then` branch or the `else` branch.
      auto* then_block = parent_node->blocks().at(0);
      auto* else_block = parent_node->blocks().at(1);

      if (parent_block == else_block) {
        // If else block then we move to the next node in the parent block.
        current_ = parent_node->next();
        if (current_->kind() == prim::Return) {
          move_up();
        }
      } else {
        // If then block then move to the else block if it is not empty.
        TORCH_INTERNAL_ASSERT(parent_block == then_block);
        bool else_block_empty =
            else_block->nodes().begin() == else_block->nodes().end();

        if (!else_block_empty) {
          current_ = *(else_block->nodes().begin());
        } else {
          // Since it's empty we move to the next node.
          current_ = parent_node->next();
          if (current_->kind() == prim::Return) {
            move_up();
          }
        }
      }
    } else if (
        parent_node->kind() == prim::Loop ||
        parent_node->kind() == prim::With) {
      current_ = parent_node->next();
      if (current_->kind() == prim::Return) {
        move_up();
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Only if/loop/with nodes should have child blocks");
    }
  }

  // Moves to the next adjacent node or up in to the parent if that is not
  // possible.
  void move_next() {
    if (current_ == nullptr) {
      return;
    }

    // Increment to the next node in the current block.
    current_ = current_->next();

    // Check if we're at the end of the block. If so we need
    // to move upwards (if it makes sense to).
    if (current_->kind() == prim::Return) {
      move_up();
    }
  }

  // Moves to the next node in the graph into children if it can.
  void move_into() {
    if (current_ == nullptr) {
      return;
    }

    // Check if we're currently on a node that contains sub-nodes.
    if (current_->kind() == prim::If || current_->kind() == prim::Loop ||
        current_->kind() == prim::With) {
      auto* first_block = current_->blocks().at(0);
      current_ = first_block->param_node();
      // Move next will move up and out of the current node if the block is
      // empty. `move_up` which is called by `move_next` will handle the
      // difference between If, Loop, and With blocks appropriately.
      move_next();
    } else {
      move_next();
    }
  }

  // Get the next Node in the graph. \returns nullptr if there are no nodes
  // left.
  Node* next() {
    auto result = current_;

    // Try move into the existing node to set the next node to be returned.
    // This will move to the next node if not possible, or move upwards and
    // to the next.
    move_into();

    return result;
  }
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `facilitates`, `DepthFirstGraphNodeIterator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/ir.h`


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

- **File Documentation**: `graph_iterator.h_docs.md`
- **Keyword Index**: `graph_iterator.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
