# Documentation: `docs/torch/csrc/jit/passes/frozen_linear_transpose.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/frozen_linear_transpose.cpp_docs.md`
- **Size**: 5,826 bytes (5.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/frozen_linear_transpose.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/frozen_linear_transpose.cpp`
- **Size**: 2,836 bytes (2.77 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/transpose.h>
#endif

#include <iostream>
#include <utility>

namespace torch::jit {
namespace {

using Tensor = at::Tensor;

class TransposeFrozenLinear {
 public:
  TransposeFrozenLinear(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run() {
    // Can't delete nodes while also iterating over it
    DepthFirstGraphNodeIterator graph_it(graph_);

    for (auto next_node = graph_it.next(); next_node != nullptr;) {
      Node* node = next_node;
      next_node = graph_it.next();

      if (is_constant_linear_op(node)) {
        replace_linear_with_matmul(node);
      }
    }
    return graph_modified_;
  }

  bool is_constant_linear_op(Node* node) {
    if (node->kind() != aten::linear) {
      return false;
    }

    // This also filters out out-variants of the linear op.
    return !nonConstantParameters(node);
  }

  void replace_linear_with_matmul(Node* node) {
    graph_modified_ = true;
    Node* matmul = nullptr;

    {
      WithInsertPoint insert_guard(node);
      auto weight = node->namedInput("weight");

      Tensor weight_tensor = constant_as<Tensor>(weight).value();
      Tensor weight_t_tensor = at::transpose(weight_tensor, 1, 0)
                                   .clone(at::MemoryFormat::Contiguous);
      Value* weight_t = graph_->insertConstant(std::move(weight_t_tensor));
      matmul = graph_->create(aten::matmul, {node->inputs()[0], weight_t});
      matmul->insertAfter(node);
    }

    // Handle a bias if there is any
    WithInsertPoint insert_guard(matmul);
    auto bias = node->namedInput("bias");
    if (bias->type() == NoneType::get()) {
      node->replaceAllUsesWith(matmul);
    } else {
      Value* bias_scale = graph_->insertConstant(1);
      Node* bias_result =
          graph_->create(aten::add, {matmul->output(), bias, bias_scale});
      bias_result->insertAfter(matmul);
      node->replaceAllUsesWith(bias_result);
    }
    node->destroy();
  }

  void handleBlockAndSubblocks(Block* block) {}

 private:
  std::shared_ptr<Graph> graph_;
  bool graph_modified_ = false;
};
} // namespace

TORCH_API bool FrozenLinearTranspose(std::shared_ptr<Graph>& graph) {
  TransposeFrozenLinear transposeWeight(graph);
  GRAPH_DUMP("Before FrozenLinearTranspose", graph);
  bool changed = transposeWeight.run();
  if (changed) {
    GRAPH_DUMP("After FrozenLinearTranspose", graph);
  }
  return changed;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `TORCH_API`

**Classes/Structs**: `TransposeFrozenLinear`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/ir_views.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/frozen_linear_transpose.h`
- `torch/csrc/jit/passes/utils/optimization_utils.h`
- `torch/csrc/jit/runtime/graph_executor.h`
- `torch/csrc/jit/runtime/graph_iterator.h`
- `ATen/Functions.h`
- `ATen/ops/transpose.h`
- `iostream`
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

- **File Documentation**: `frozen_linear_transpose.cpp_docs.md`
- **Keyword Index**: `frozen_linear_transpose.cpp_kw.md`
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

- **File Documentation**: `frozen_linear_transpose.cpp_docs.md_docs.md`
- **Keyword Index**: `frozen_linear_transpose.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
