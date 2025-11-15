# Documentation: `torch/csrc/jit/passes/requires_grad_analysis.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/requires_grad_analysis.cpp`
- **Size**: 5,270 bytes (5.15 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/requires_grad_analysis.h>

#include <ATen/core/jit_type.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>

#include <vector>

namespace torch::jit {

namespace {

bool getRequiresGrad(Value* value) {
  return value->requires_grad();
}

void setRequiresGrad(Value* value, bool req_value) {
  if (auto type = value->type()->cast<TensorType>()) {
    value->setType(type->withRequiresGrad(req_value));
  }
}

void setRequiresGrad(
    at::ArrayRef<Value*> outputs,
    const std::vector<bool>& values) {
  AT_ASSERT(outputs.size() == values.size());
  for (const auto i : c10::irange(values.size())) {
    setRequiresGrad(outputs[i], values[i]);
  }
}

void setRequiresGrad(Node* node, const std::vector<bool>& values) {
  setRequiresGrad(node->outputs(), values);
}

std::vector<bool> bitwiseOr(std::vector<bool> a, const std::vector<bool>& b) {
  AT_ASSERT(a.size() == b.size());
  for (const auto i : c10::irange(a.size())) {
    a[i] = a[i] || b[i];
  }
  return a;
}

void PropagateRequiresGradSimpleNode(Node* node) {
  static const OperatorSet comparison_ops = {
      "aten::lt(Tensor self, Tensor other) -> Tensor",
      "aten::le(Tensor self, Tensor other) -> Tensor",
      "aten::gt(Tensor self, Tensor other) -> Tensor",
      "aten::ge(Tensor self, Tensor other) -> Tensor",
      "aten::eq(Tensor self, Tensor other) -> Tensor",
      "aten::ne(Tensor self, Tensor other) -> Tensor",
      "aten::lt(Tensor self, Scalar other) -> Tensor",
      "aten::le(Tensor self, Scalar other) -> Tensor",
      "aten::gt(Tensor self, Scalar other) -> Tensor",
      "aten::ge(Tensor self, Scalar other) -> Tensor",
      "aten::eq(Tensor self, Scalar other) -> Tensor",
      "aten::ne(Tensor self, Scalar other) -> Tensor",
  };

  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (node->isMemberOf(comparison_ops)) {
    return setRequiresGrad(node->output(), false);
  } else if (node->matches(
                 "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
    return setRequiresGrad(node->output(), node->input(0)->requires_grad());
  } else if (node->matches("aten::detach(Tensor(a) self) -> Tensor(a)")) {
    return setRequiresGrad(node->output(), false);
  } else if (node->kind() == aten::tensor) {
    if (auto grad_index =
            node->schema().argumentIndexWithName("requires_grad")) {
      if (auto const_arg = constant_as<bool>(node->inputs().at(*grad_index))) {
        return setRequiresGrad(node->output(), *const_arg);
      }
    }
    if (auto type = node->output()->type()->cast<TensorType>()) {
      if (type->scalarType()) {
        setRequiresGrad(
            node->output(),
            autograd::isDifferentiableType(*type->scalarType()));
      }
    }
    return;
  }

  auto inputs = node->inputs();
  auto outputs = node->outputs();
  bool should_require =
      std::any_of(inputs.begin(), inputs.end(), getRequiresGrad);
  for (Value* output : outputs) {
    if (auto type = output->type()->cast<TensorType>()) {
      if (type->scalarType()) {
        setRequiresGrad(
            output,
            should_require &&
                autograd::isDifferentiableType(*type->scalarType()));
      }
    }
  }
}

void PropagateRequiresGrad(Block* block);

void PropagateRequiresGrad(Node* node) {
  if (node->kind() == prim::If) {
    auto blocks = node->blocks();
    auto true_block = blocks.at(0);
    auto false_block = blocks.at(1);

    PropagateRequiresGrad(true_block);
    PropagateRequiresGrad(false_block);

    auto outputs_require = bitwiseOr(
        fmap(true_block->outputs(), getRequiresGrad),
        fmap(false_block->outputs(), getRequiresGrad));
    setRequiresGrad(node, outputs_require);
  } else if (node->kind() == prim::Loop) {
    auto body = node->blocks().at(0);
    std::vector<bool> loop_inputs_require =
        fmap(node->inputs().slice(2), getRequiresGrad);
    std::vector<bool> body_inputs_require = loop_inputs_require;
    std::vector<bool> body_outputs_require(node->outputs().size(), false);

    std::vector<bool> new_body_inputs_require = body_inputs_require;
    std::vector<bool> new_body_outputs_require = body_outputs_require;

    // continue iterating until the results have converged
    do {
      body_inputs_require = new_body_inputs_require;
      body_outputs_require = new_body_outputs_require;

      new_body_inputs_require =
          bitwiseOr(body_inputs_require, body_outputs_require);
      setRequiresGrad(
          body->param_node()->outputs().slice(1), new_body_inputs_require);
      PropagateRequiresGrad(body);
      new_body_outputs_require =
          fmap(body->return_node()->inputs().slice(1), getRequiresGrad);
    } while (new_body_inputs_require != body_inputs_require ||
             new_body_outputs_require != body_outputs_require);

    setRequiresGrad(node, bitwiseOr(body_outputs_require, loop_inputs_require));
  } else {
    PropagateRequiresGradSimpleNode(node);
  }
}

void PropagateRequiresGrad(Block* block) {
  for (Node* node : block->nodes()) {
    PropagateRequiresGrad(node);
  }
}
} // anonymous namespace

void PropagateRequiresGrad(std::shared_ptr<Graph>& graph) {
  PropagateRequiresGrad(graph->block());
}
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

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

- `torch/csrc/jit/passes/requires_grad_analysis.h`
- `ATen/core/jit_type.h`
- `c10/util/irange.h`
- `torch/csrc/jit/ir/constants.h`
- `torch/csrc/jit/ir/ir.h`
- `vector`


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

- **File Documentation**: `requires_grad_analysis.cpp_docs.md`
- **Keyword Index**: `requires_grad_analysis.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
