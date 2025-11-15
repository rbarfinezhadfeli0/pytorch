# Documentation: `docs/torch/csrc/jit/passes/restore_mutation.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/restore_mutation.cpp_docs.md`
- **Size**: 5,382 bytes (5.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/restore_mutation.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/restore_mutation.cpp`
- **Size**: 2,695 bytes (2.63 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>

namespace torch::jit {

FunctionalToInplaceRewriter::FunctionalToInplaceRewriter(
    std::shared_ptr<Graph> graph)
    : aliasDb_(nullptr), graph_(std::move(graph)) {}

bool FunctionalToInplaceRewriter::CanBeInplace(Node* node) {
  if (activation_type_promotion_mapping.find(node->kind()) ==
      activation_type_promotion_mapping.end()) {
    return false;
  }

  Symbol inplace_op =
      Symbol::fromQualString(std::string(node->kind().toQualString()) + "_");
  if (!inplace_op) {
    return false;
  }

  // If type promotion is allowed, then perform dtype check
  bool check_dtype = activation_type_promotion_mapping.at(node->kind());

  Value* input = node->inputs().at(0);
  Value* output = node->outputs().at(0);
  auto inputDtype = input->type()->expect<TensorType>()->scalarType();
  auto outputDtype = output->type()->expect<TensorType>()->scalarType();

  // In general, we don't need to check shape for activation ops as they
  // element-wise. But for those where type promotion could happen, we need to
  // make sure the dtype of input and output are the same. For now the dtype
  // checking will always fail until the type inference is ready.
  if (check_dtype &&
      (!inputDtype || !outputDtype ||
       inputDtype.value() != outputDtype.value())) {
    return false;
  }

  // Skip if input's def node has side effect or input has alias
  if (MutationRemover::hasSideEffectOrAlias(input, getOrCreateAliasDb())) {
    return false;
  }

  // If x has more than one use, skip the conversion.
  // TODO: Use liveness analysis to catch more general scenario
  return (input->uses().size() == 1);
}

bool FunctionalToInplaceRewriter::FunctionalToInplace(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    for (Block* sub_block : node->blocks()) {
      changed |= FunctionalToInplace(sub_block);
    }

    if (!CanBeInplace(node)) {
      continue;
    }

    changed = true;
    Node* inplace_node = node->replaceWithNewSymbol(
        Symbol::fromQualString(node->schema().name() + "_"));
    inplace_node->output()->replaceAllUsesWith(node->inputs().at(0));
    getOrCreateAliasDb()->replaceWithNewValue(
        node->output(), inplace_node->output());

    node->destroy();
  }
  return changed;
}

bool FunctionalToInplaceActivation(const std::shared_ptr<Graph>& graph) {
  FunctionalToInplaceRewriter rewriter(graph);
  return rewriter.FunctionalToInplace(graph->block());
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/jit_type.h`
- `ATen/core/symbol.h`
- `torch/csrc/jit/passes/remove_mutation.h`
- `torch/csrc/jit/passes/restore_mutation.h`


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

- **File Documentation**: `restore_mutation.cpp_docs.md`
- **Keyword Index**: `restore_mutation.cpp_kw.md`
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

- **File Documentation**: `restore_mutation.cpp_docs.md_docs.md`
- **Keyword Index**: `restore_mutation.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
