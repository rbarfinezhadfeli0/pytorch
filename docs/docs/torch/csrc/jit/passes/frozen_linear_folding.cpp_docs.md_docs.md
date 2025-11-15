# Documentation: `docs/torch/csrc/jit/passes/frozen_linear_folding.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/frozen_linear_folding.cpp_docs.md`
- **Size**: 8,044 bytes (7.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/frozen_linear_folding.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/frozen_linear_folding.cpp`
- **Size**: 5,065 bytes (4.95 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_linear_bn.h>
#include <torch/csrc/jit/passes/frozen_linear_folding.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace torch::jit {

namespace {

using Tensor = at::Tensor;

bool supportedLinearNode(Node* n) {
  if (n->kind() == aten::linear) {
    return true;
  } else {
    return false;
  }
}

bool FoldFrozenLinearBatchnorm(Block* b) {
  bool graph_modified = false;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      graph_modified |= FoldFrozenLinearBatchnorm(block);
    }

    if (n->kind() == aten::batch_norm &&
        supportedLinearNode(n->inputs().at(0)->node())) {
      auto linear = n->inputs().at(0)->node();
      auto bn = n;

      if (nonConstantParameters(linear) || nonConstantParameters(bn)) {
        continue;
      }

      auto bn_rm_ivalue = bn->namedInput("running_mean");
      auto bn_rv_ivalue = bn->namedInput("running_var");

      // check running_mean and running_var has value, if they are
      // None(track_running_stats=False), skipping the folding path.
      if (bn_rm_ivalue->type() == NoneType::get() &&
          bn_rv_ivalue->type() == NoneType::get()) {
        continue;
      }

      auto bn_rm = constant_as<Tensor>(bn->namedInput("running_mean")).value();
      auto bn_rv = constant_as<Tensor>(bn->namedInput("running_var")).value();
      auto bn_eps = constant_as<double>(bn->namedInput("eps")).value();
      auto linear_w = constant_as<Tensor>(linear->namedInput("weight")).value();

      int64_t linear_out_features = linear_w.size(0);
      int64_t bn_num_features = bn_rm.size(0);

      // Linear-BN needs to be fused while preserving the shapes of linear
      // weight/bias. To preserve the shapes of linear weight/bias, the channel
      // dim of bn needs to be broadcastable with the last dim of linear,
      // because bn operates over the channel dim, (N, C_in, H, W) while linear
      // operates over the last dim, (*, H_in). To be broadcastable, the number
      // of features in bn and the number of output features from linear must
      // satisfy the following condition:
      // 1. they are equal, or
      // 2. the number of features in bn is 1
      // Otherwise, skip the folding path
      if (!(linear_out_features == bn_num_features || bn_num_features == 1)) {
        continue;
      }

      // implementation taken from torch/nn/utils/fusion.py
      Tensor linear_b;
      if (linear->namedInput("bias")->type() == NoneType::get()) {
        at::ScalarType bias_dtype = bn_rm.scalar_type();
        at::ScalarType weight_dtype = linear_w.scalar_type();
        at::DeviceType weight_device = linear_w.device().type();
        if (weight_device == at::kCUDA &&
            (weight_dtype == at::kHalf || weight_dtype == at::kBFloat16) &&
            bias_dtype == at::kFloat) {
          bias_dtype = weight_dtype;
        }
        linear_b = at::zeros_like(bn_rm, at::TensorOptions().dtype(bias_dtype));
      } else {
        linear_b = constant_as<Tensor>(linear->namedInput("bias")).value();
      }
      Tensor bn_w;
      if (bn->namedInput("weight")->type() == NoneType::get()) {
        bn_w = at::ones_like(bn_rm);
      } else {
        bn_w = constant_as<Tensor>(bn->namedInput("weight")).value();
      }
      Tensor bn_b;
      if (n->namedInput("bias")->type() == NoneType::get()) {
        bn_b = at::zeros_like(bn_rm);
      } else {
        bn_b = constant_as<Tensor>(bn->namedInput("bias")).value();
      }

      LinearBNParameters params;
      params.linear_w = linear_w;
      params.linear_b = linear_b;
      params.bn_rm = bn_rm;
      params.bn_rv = bn_rv;
      params.bn_eps = bn_eps;
      params.bn_w = bn_w;
      params.bn_b = bn_b;
      std::tuple<Tensor, Tensor> out =
          computeUpdatedLinearWeightAndBias(params);
      WithInsertPoint guard(linear);
      auto fused_linear_w = b->owningGraph()->insertConstant(std::get<0>(out));
      auto fused_linear_b = b->owningGraph()->insertConstant(std::get<1>(out));
      auto linear_w_value = linear->namedInput("weight");
      auto linear_b_value = linear->namedInput("bias");

      fused_linear_w->setDebugName(linear_w_value->debugName() + "_fused_bn");
      fused_linear_b->setDebugName(linear_b_value->debugName() + "_fused_bn");

      linear->replaceInputWith(linear_w_value, fused_linear_w);
      linear->replaceInputWith(linear_b_value, fused_linear_b);

      bn->output()->replaceAllUsesWith(linear->output());
      graph_modified = true;
    }
  }
  return graph_modified;
}

} // namespace

bool FoldFrozenLinearBatchnorm(std::shared_ptr<Graph>& graph) {
  bool graph_modified = FoldFrozenLinearBatchnorm(graph->block());
  EliminateDeadCode(graph);
  return graph_modified;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `bool`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/constants.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/passes/dead_code_elimination.h`
- `torch/csrc/jit/passes/fold_linear_bn.h`
- `torch/csrc/jit/passes/frozen_linear_folding.h`
- `torch/csrc/jit/passes/utils/optimization_utils.h`
- `ATen/Functions.h`
- `ATen/ops/ones_like.h`
- `ATen/ops/zeros_like.h`


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

- **File Documentation**: `frozen_linear_folding.cpp_docs.md`
- **Keyword Index**: `frozen_linear_folding.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `frozen_linear_folding.cpp_docs.md_docs.md`
- **Keyword Index**: `frozen_linear_folding.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
