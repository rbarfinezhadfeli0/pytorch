# Documentation: `docs/torch/csrc/jit/passes/prepack_folding.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/prepack_folding.cpp_docs.md`
- **Size**: 5,151 bytes (5.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/prepack_folding.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/prepack_folding.cpp`
- **Size**: 2,425 bytes (2.37 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <stack>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/prepack_folding.h>

namespace torch::jit {

// Must run this pass after constant folding.
void PrePackingOpsFolder(
    script::Module& m,
    const PrePackingOpsFilterFn& is_foldable_op,
    const std::string& attr_prefix) {
  for (auto& method : m.get_methods()) {
    int64_t uid = 0; // int + method name gives unique identifier
    auto graph = method.graph();
    std::stack<Block*> blocks_to_visit;
    std::unordered_set<Node*> nodes_to_delete;
    blocks_to_visit.push(graph->block());
    std::string attr_name_base =
        attr_prefix + "_" + method.name() + "._jit_pass_packed_weight_";
    while (!blocks_to_visit.empty()) {
      Block* b = blocks_to_visit.top();
      blocks_to_visit.pop();
      for (Node* n : b->nodes()) {
        if (is_foldable_op(n)) {
          auto optional_outputs = runNodeIfInputsAreConstant(n);
          if (optional_outputs) {
            auto outputs = optional_outputs.value();
            TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");
            auto attr_name = attr_name_base + std::to_string(uid++);
            TORCH_CHECK(
                !(m.type()->findAttributeSlot(attr_name)),
                "Attribute name ",
                attr_name,
                " already exist in",
                " module of type:",
                m.type()->name()->qualifiedName(),
                ". Please make sure that",
                " FoldPrePackingOps is run at the top level module only.");
            m.register_attribute(attr_name, n->output(0)->type(), outputs[0]);
            Value* prepack_op_value = n->output(0);
            WithInsertPoint ins(prepack_op_value->node());
            Value* packed_weight_attr =
                graph->insertGetAttr(graph->inputs()[0], attr_name)
                    ->setType(n->output(0)->type());
            prepack_op_value->replaceAllUsesWith(packed_weight_attr);
            nodes_to_delete.insert(n);
          }
        }
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
        }
      }
    }
    for (auto n : nodes_to_delete) {
      n->removeAllInputs();
    }
    for (auto n : nodes_to_delete) {
      n->destroy();
    }
  }
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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

- `stack`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/passes/constant_pooling.h`
- `torch/csrc/jit/passes/constant_propagation.h`
- `torch/csrc/jit/passes/prepack_folding.h`


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

- **File Documentation**: `prepack_folding.cpp_docs.md`
- **Keyword Index**: `prepack_folding.cpp_kw.md`
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

- **File Documentation**: `prepack_folding.cpp_docs.md_docs.md`
- **Keyword Index**: `prepack_folding.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
