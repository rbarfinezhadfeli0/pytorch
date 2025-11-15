# Documentation: `docs/torch/csrc/jit/passes/constant_pooling.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/constant_pooling.cpp_docs.md`
- **Size**: 4,968 bytes (4.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/constant_pooling.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/constant_pooling.cpp`
- **Size**: 2,221 bytes (2.17 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/constant_pooling.h>

#include <ATen/core/symbol.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <unordered_set>

namespace torch::jit {

namespace {

// Very similar to the common subexpression elimination pass
// Move all constants to the beginning of the graph, and deduplicate
void ConstantPooling(
    Block* block,
    std::unordered_set<Node*, HashNode, EqualNode>& constants,
    const AliasDb& aliasDb) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto node = *it;
    // node may be moved to a different block so advance iterator now
    ++it;
    if (!node->blocks().empty()) {
      // Traverse sub-blocks.
      for (auto block : node->blocks()) {
        ConstantPooling(block, constants, aliasDb);
      }
      continue;
    }

    if (node->kind() != prim::Constant) {
      continue;
    }

    // Check whether the same constant already exists.
    auto subit = constants.insert(node);
    if (!subit.second) {
      auto existing = *subit.first;

      auto old_ivalue = toIValue(existing->output());
      auto new_ivalue = toIValue(node->output());

      // if both values are the same object, we do not need to worry about
      // changing the aliasing relationship
      bool same_identity =
          (old_ivalue && new_ivalue && (old_ivalue->is(new_ivalue)));

      if (!same_identity &&
          !aliasDb.safeToChangeAliasingRelationship(
              node->outputs(), existing->outputs())) {
        continue;
      }

      // constant exists, replace the uses of node, and destroy it.
      node->replaceAllUsesWith(existing);
      node->destroy();
      continue;
    }

    // Move the constant definition to the beginning of the graph.
    auto first_node = node->owningGraph()->block()->nodes().front();
    if (node != first_node)
      node->moveBefore(first_node);
  }
}
} // anonymous namespace

void ConstantPooling(const std::shared_ptr<Graph>& graph) {
  AliasDb aliasDb(graph);
  std::unordered_set<Node*, HashNode, EqualNode> constants;
  ConstantPooling(graph->block(), constants, aliasDb);
}
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `torch/csrc/jit/passes/constant_pooling.h`
- `ATen/core/symbol.h`
- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/node_hashing.h`
- `unordered_set`


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

- **File Documentation**: `constant_pooling.cpp_docs.md`
- **Keyword Index**: `constant_pooling.cpp_kw.md`
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

- **File Documentation**: `constant_pooling.cpp_docs.md_docs.md`
- **Keyword Index**: `constant_pooling.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
