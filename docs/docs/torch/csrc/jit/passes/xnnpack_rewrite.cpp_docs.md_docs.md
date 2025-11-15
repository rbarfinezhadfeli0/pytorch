# Documentation: `docs/torch/csrc/jit/passes/xnnpack_rewrite.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/xnnpack_rewrite.cpp_docs.md`
- **Size**: 23,192 bytes (22.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/xnnpack_rewrite.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/xnnpack_rewrite.cpp`
- **Size**: 19,947 bytes (19.48 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/jit_type.h>
#include <ATen/native/xnnpack/OpContext.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/hoist_conv_packed_params.h>
#include <torch/csrc/jit/passes/mobile_optimizer_type.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch::jit {

namespace {

void replaceConv1dWithConv2d(std::shared_ptr<Graph>& graph) {
  std::string conv_1d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::conv1d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res) )";

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %zero : int = prim::Constant[value=0]()
        %one : int = prim::Constant[value=1]()
        %stride_w : int = prim::ListUnpack(%stride)
        %stride_2d : int[] = prim::ListConstruct(%one, %stride_w)
        %padding_w : int = prim::ListUnpack(%padding)
        %padding_2d : int[] = prim::ListConstruct(%zero, %padding_w)
        %dilation_w : int = prim::ListUnpack(%dilation)
        %dilation_2d : int[] = prim::ListConstruct(%one, %dilation_w)
        %two : int = prim::Constant[value=2]()
        %input_2d : Tensor = aten::unsqueeze(%input, %two)
        %weight_2d : Tensor = aten::unsqueeze(%weight, %two)
        %output_2d = aten::conv2d(
            %input_2d, %weight_2d, %bias, %stride_2d, %padding_2d, %dilation_2d, %groups)
        %output : Tensor = aten::squeeze(%output_2d, %two)
        return (%output) )";

  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"zero", "res"},
       {"one", "res"},
       {"stride_w", "res"},
       {"stride_2d", "res"},
       {"padding_w", "res"},
       {"padding_2d", "res"},
       {"dilation_w", "res"},
       {"dilation_2d", "res"},
       {"two", "res"},
       {"input_2d", "res"},
       {"weight_2d", "res"},
       {"output_2d", "res"},
       {"output", "res"}});

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_1d_pattern, conv_2d_pattern, value_mappings);
  rewriter.runOnGraph(graph);
}

} // namespace

void transformConv1dToConv2d(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv1d and conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);
  replaceConv1dWithConv2d(graph);
}

void transformConv1dToConv2d(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    transformConv1dToConv2d(graph);
  }
  for (script::Module m : module.children()) {
    transformConv1dToConv2d(m);
  }
}

#ifdef USE_XNNPACK

namespace {

void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  // fuse decomposed linear into aten::linear
  FuseLinear(graph);

  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %res = aten::linear(%input, %weight, %bias)
        return (%res))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"output_min_max", "res"},
       {"packed_weight_bias", "res"},
       {"res", "res"}});

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(
      linear_pattern, prepacked_ops_pattern, value_mappings);
  linear_rewriter.runOnGraph(graph);
}

void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%res) )";

  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"output_min_max", "res"},
       {"packed_weight_bias", "res"},
       {"res", "res"}});

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern, value_mappings);
  rewriter.runOnGraph(graph);

  std::string conv_2d_transpose_pattern = R"(
      graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[],
          %output_padding:int[], %groups:int):
        %res = aten::conv_transpose2d(%input, %weight, %bias, %stride, %padding, %output_padding, %groups, %dilation)
        return (%res) )";

  std::string prepacked_ops_conv2d_transpose_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %output_padding:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::conv2d_transpose_clamp_prepack(
            %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %res = prepacked::conv2d_transpose_clamp_run(%input, %packed_weight_bias)
        return (%res) )";

  value_mappings = {
      {"output_min_max", "res"}, {"packed_weight_bias", "res"}, {"res", "res"}};

  SubgraphRewriter transpose_rewriter;
  transpose_rewriter.RegisterRewritePattern(
      conv_2d_transpose_pattern,
      prepacked_ops_conv2d_transpose_pattern,
      value_mappings);
  transpose_rewriter.runOnGraph(graph);
}

void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string linear_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.xnnpack.LinearOpContext = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.xnnpack.Conv2dOpContext = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%res) )";

  std::string linear_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh(%linear_res, %output_min, %output_max)
        return (%res))";

  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"packed_weight_bias", "packed_weight_bias"}, {"res", "res"}});

  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh,
      linear_prepack_run_hardtanh_fused,
      value_mappings);

  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh(%conv2d_res, %output_min, %output_max)
        return (%res) )";

  value_mappings = {
      {"packed_weight_bias", "packed_weight_bias"}, {"res", "res"}};

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh,
      conv2d_prepack_run_hardtanh_fused,
      value_mappings);

  std::string linear_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh_(%linear_res, %output_min, %output_max)
        return (%res))";

  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh_(%conv2d_res, %output_min, %output_max)
        return (%res) )";

  value_mappings = {
      {"packed_weight_bias", "packed_weight_bias"}, {"res", "res"}};

  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh_inplace,
      linear_prepack_run_hardtanh_fused,
      value_mappings);

  value_mappings = {
      {"packed_weight_bias", "packed_weight_bias"}, {"res", "res"}};

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh_inplace,
      conv2d_prepack_run_hardtanh_fused,
      value_mappings);

  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string linear_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.xnnpack.LinearOpContext = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  std::string conv2d_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.xnnpack.Conv2dOpContext = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%res) )";

  std::string linear_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu(%linear_res)
        return (%res))";

  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"output_min", "packed_weight_bias"},
       {"output_max", "packed_weight_bias"},
       {"packed_weight_bias", "packed_weight_bias"},
       {"res", "res"}});

  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu, linear_prepack_run_relu_fused, value_mappings);

  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu(%conv2d_res)
        return (%res) )";

  value_mappings = {
      {"output_min", "packed_weight_bias"},
      {"output_max", "packed_weight_bias"},
      {"packed_weight_bias", "packed_weight_bias"},
      {"res", "res"}};

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused, value_mappings);

  std::string linear_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu_(%linear_res)
        return (%res))";

  std::string conv2d_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu_(%conv2d_res)
        return (%res) )";

  value_mappings = {
      {"output_min", "packed_weight_bias"},
      {"output_max", "packed_weight_bias"},
      {"packed_weight_bias", "packed_weight_bias"},
      {"res", "res"}};

  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu_inplace,
      linear_prepack_run_relu_fused,
      value_mappings);

  value_mappings = {
      {"output_min", "packed_weight_bias"},
      {"output_max", "packed_weight_bias"},
      {"packed_weight_bias", "packed_weight_bias"},
      {"res", "res"}};

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu_inplace,
      conv2d_prepack_run_relu_fused,
      value_mappings);
  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

void runCanonicalOptimizations(script::Module& module) {
  for (const auto& method : module.get_methods()) {
    auto graph = method.graph();
    // Not sure if we have models running on mobile that require loop unrolling.
    // Perhaps language/speech models? Conservatively setting that to false.
    runOptimization(graph, false /* no loop unrolling */);
  }
}

} // namespace

void insertPrePackedOps(std::shared_ptr<Graph>& graph) {
  insertPrePackedLinearOp(graph);
  insertPrePackedConv2dOp(graph);
}

void insertPrePackedOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    insertPrePackedOps(graph);
  }
  for (script::Module m : module.children()) {
    insertPrePackedOps(m);
  }
}

void fusePrePackedLinearConvWithClamp(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    fuseReluWithPackedOps(graph);
    fuseHardtanhWithPackedOps(graph);

    // Ignore user defined classes for later passes
    ConstantPropagation(graph, true);
  }
}

void FoldPrePackingOps(script::Module& m) {
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() ==
         Symbol::fromQualString("prepacked::linear_clamp_prepack")) ||
        n->kind() ==
            Symbol::fromQualString("prepacked::conv2d_clamp_prepack") ||
        n->kind() ==
            Symbol::fromQualString(
                "prepacked::conv2d_transpose_clamp_prepack"));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
  for (auto& method : m.get_methods()) {
    auto graph = method.graph();
    // Folding requires a const propagation through user defined classes
    ConstantPropagation(graph, false);
  }
}

script::Module optimizeForMobile(
    const script::Module& m,
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods) {
  auto cloned_module = m.clone();
  cloned_module.eval();

  if (!optimization_blocklist.count(MobileOptimizerType::CONV_1D_TO_2D)) {
    transformConv1dToConv2d(cloned_module);
  }

  if (!optimization_blocklist.count(MobileOptimizerType::CONV_BN_FUSION)) {
    cloned_module = FoldConvBatchNorm(cloned_module);
  }

  // Many optimizations require a frozen module, but ConvBatchNorm requires
  // an unfrozen module
  cloned_module = freeze_module(cloned_module, preserved_methods);

  if (!optimization_blocklist.count(
          MobileOptimizerType::INSERT_FOLD_PREPACK_OPS)) {
    // TODO fix duplication caused by referencing same op across multiple
    // functions
    insertPrePackedOps(cloned_module);
    cloned_module = freeze_module(cloned_module, preserved_methods);
    fusePrePackedLinearConvWithClamp(cloned_module);
    FoldPrePackingOps(cloned_module);
  }

  if (!optimization_blocklist.count(
          MobileOptimizerType::HOIST_CONV_PACKED_PARAMS) &&
      cloned_module.find_method("forward")) {
    // freeze again in case it was not done in previous optional passes
    cloned_module = freeze_module(cloned_module, preserved_methods);
    HoistConvPackedParams(cloned_module);
    // and freeze yet again to remove the empty QuantizedConv modules
    cloned_module = freeze_module(cloned_module, preserved_methods);
  }

  // Run canonical optimizations post freezing
  // since freezing inlines the graph. Otherwise we
  // will have to explicitly call Inlining pass.
  runCanonicalOptimizations(cloned_module);

  if (!optimization_blocklist.count(MobileOptimizerType::REMOVE_DROPOUT)) {
    for (const auto& method : cloned_module.get_methods()) {
      auto graph = method.graph();
      // Module must be not be in training mode but optimize calls eval()
      removeDropout(graph);
    }
  }

  if (!optimization_blocklist.count(MobileOptimizerType::FUSE_ADD_RELU)) {
    for (const auto& method : cloned_module.get_methods()) {
      auto graph = method.graph();
      FuseAddRelu(graph);
    }
  }
  cloned_module.register_attribute("mobile_optimized", BoolType::get(), true);
  return cloned_module;
}

#else

void insertPrePackedOps(std::shared_ptr<Graph>& graph) {
  TORCH_INTERNAL_ASSERT(
      false, "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void insertPrePackedOps(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      false, "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void fusePrePackedLinearConvWithClamp(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      false, "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void FoldPrePackingOps(script::Module& m) {
  TORCH_INTERNAL_ASSERT(
      false, "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

script::Module optimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& blocklist,
    const std::vector<std::string>& preserved_methods) {
  TORCH_INTERNAL_ASSERT(
      false,
      "Mobile optimization only available with XNNPACK at the moment. "
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
  return module;
}

#endif
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 23 function(s).

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
- `ATen/native/xnnpack/OpContext.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/passes/constant_propagation.h`
- `torch/csrc/jit/passes/fold_conv_bn.h`
- `torch/csrc/jit/passes/freeze_module.h`
- `torch/csrc/jit/passes/fuse_linear.h`
- `torch/csrc/jit/passes/fuse_relu.h`
- `torch/csrc/jit/passes/graph_rewrite_helper.h`
- `torch/csrc/jit/passes/hoist_conv_packed_params.h`
- `torch/csrc/jit/passes/mobile_optimizer_type.h`
- `torch/csrc/jit/passes/prepack_folding.h`
- `torch/csrc/jit/passes/remove_dropout.h`
- `torch/csrc/jit/passes/subgraph_rewrite.h`
- `torch/csrc/jit/passes/xnnpack_rewrite.h`
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

- **File Documentation**: `xnnpack_rewrite.cpp_docs.md`
- **Keyword Index**: `xnnpack_rewrite.cpp_kw.md`
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

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

- **File Documentation**: `xnnpack_rewrite.cpp_docs.md_docs.md`
- **Keyword Index**: `xnnpack_rewrite.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
