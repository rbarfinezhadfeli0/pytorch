# Documentation: `torch/csrc/jit/passes/metal_rewrite.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/metal_rewrite.cpp`
- **Size**: 11,170 bytes (10.91 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/jit_type.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/metal_rewrite.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch::jit {

namespace {

void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  // fuse decomposed linear into aten::linear
  FuseLinear(graph);

  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %res = metal_prepack::linear_run(%input, %packed_weight_bias)
        return (%res))";

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, prepacked_ops_pattern);
  linear_rewriter.runOnGraph(graph);
}

void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern);
  rewriter.runOnGraph(graph);
}

void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string linear_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.metal.LinearOpContext = metal_prepack::linear_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = metal_prepack::linear_run(%input, %packed_weight_bias)
        return (%res))";

  std::string linear_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = metal_prepack::linear_run(%input, %packed_weight_bias)
        %res = aten::relu(%linear_res)
        return (%res))";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu, linear_prepack_run_relu_fused);

  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        %r = aten::relu(%r)
        return (%r) )";

  std::string conv2d_prepack_run_relu_fused = R"(
  graph(%input, %weight, %bias, %stride:int[], %padding:int[],
        %dilation:int[], %groups:int, %dummy_min_max):
      %output_min: float = prim::Constant[value=0.0]()
      %output_max: None = prim::Constant()
      %packed_weight_bias: __torch__.torch.classes.metal.Conv2dOpContext = metal_prepack::conv2d_prepack(
          %weight, %bias, %stride, %padding, %dilation, %groups,
          %output_min, %output_max)
      %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
      return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused);

  std::string linear_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = metal_prepack::linear_run(%input, %packed_weight_bias)
        %res = aten::relu_(%linear_res)
        return (%res))";

  std::string conv2d_prepack_run_relu_inplace = R"(
  graph(%input, %weight, %bias, %stride:int[], %padding:int[],
        %dilation:int[], %groups:int, %dummy_min_max):
      %packed_weight_bias = metal_prepack::conv2d_prepack(
          %weight, %bias, %stride, %padding, %dilation, %groups,
          %dummy_min_max, %dummy_min_max)
      %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
      %r = aten::relu_(%r)
      return (%r) )";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu_inplace, linear_prepack_run_relu_fused);
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu_inplace, conv2d_prepack_run_relu_fused);

  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string linear_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.metal.LinearOpContext = metal_prepack::linear_prepack(%weight, %bias, %output_min, %output_max)
        %res = metal_prepack::linear_run(%input, %packed_weight_bias)
        return (%res))";

  std::string linear_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = metal_prepack::linear_run(%input, %packed_weight_bias)
        %res = aten::hardtanh(%linear_res, %output_min, %output_max)
        return (%res))";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh, linear_prepack_run_hardtanh_fused);

  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias: __torch__.torch.classes.metal.Conv2dOpContext = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        return (%r) )";

  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        %r = aten::hardtanh(%r, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh, conv2d_prepack_run_hardtanh_fused);

  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        %r = aten::hardtanh_(%r, %output_min, %output_max)
        return (%r) )";

  std::string linear_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = metal_prepack::linear_run(%input, %packed_weight_bias)
        %res = aten::hardtanh_(%linear_res, %output_min, %output_max)
        return (%res))";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh_inplace, linear_prepack_run_hardtanh_fused);

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh_inplace, conv2d_prepack_run_hardtanh_fused);

  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

} // namespace

void metalInsertPrePackedOps(std::shared_ptr<Graph>& graph) {
  insertPrePackedLinearOp(graph);
  insertPrePackedConv2dOp(graph);
}

void metalInsertPrePackedOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    metalInsertPrePackedOps(graph);
  }
  for (script::Module m : module.children()) {
    metalInsertPrePackedOps(m);
  }
}

void metalFoldPrePackingOps(script::Module& m) {
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() ==
         Symbol::fromQualString("metal_prepack::conv2d_prepack")) ||
        (n->kind() == Symbol::fromQualString("metal_prepack::linear_prepack")));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
}

void metalFusePrePackedConvWithClamp(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  fuseReluWithPackedOps(graph);
  fuseHardtanhWithPackedOps(graph);
}

static void metalRemoveMutation(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  RemoveTensorMutation(graph);
}

static void metalRunCanonicalOptimizations(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  runOptimization(graph, false /* no loop unrolling */);
}

script::Module metalOptimizeForMobile(
    const script::Module& m,
    const std::vector<std::string>& preserved_methods) {
  auto cloned_module = m.clone();
  cloned_module.eval();
  cloned_module = FoldConvBatchNorm(cloned_module);
  metalInsertPrePackedOps(cloned_module);
  cloned_module = freeze_module(cloned_module, preserved_methods);
  metalFusePrePackedConvWithClamp(cloned_module);
  metalFoldPrePackingOps(cloned_module);
  removeDropout(cloned_module);
  metalRemoveMutation(cloned_module);
  // remove duplicated constants
  metalRunCanonicalOptimizations(cloned_module);
  cloned_module.register_attribute(
      "optimized_for_metal", BoolType::get(), true);
  return cloned_module;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

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

- `ATen/core/jit_type.h`
- `c10/util/irange.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/subgraph_matcher.h`
- `torch/csrc/jit/passes/constant_pooling.h`
- `torch/csrc/jit/passes/fold_conv_bn.h`
- `torch/csrc/jit/passes/freeze_module.h`
- `torch/csrc/jit/passes/fuse_linear.h`
- `torch/csrc/jit/passes/graph_rewrite_helper.h`
- `torch/csrc/jit/passes/metal_rewrite.h`
- `torch/csrc/jit/passes/prepack_folding.h`
- `torch/csrc/jit/passes/remove_dropout.h`
- `torch/csrc/jit/passes/remove_mutation.h`
- `torch/csrc/jit/passes/subgraph_rewrite.h`
- `torch/csrc/jit/runtime/graph_executor_impl.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

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

- **File Documentation**: `metal_rewrite.cpp_docs.md`
- **Keyword Index**: `metal_rewrite.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
