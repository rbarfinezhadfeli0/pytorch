# Documentation: `docs/torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp_docs.md`
- **Size**: 8,077 bytes (7.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp`
- **Size**: 4,997 bytes (4.88 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Utils.h>

#include <ATen/code_template.h>
#include <ATen/cuda/CUDAConfig.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch::jit {

namespace {
void fuseFrozenConvAddReluImpl(std::shared_ptr<Graph>& graph) {
#if AT_CUDNN_ENABLED() || AT_ROCM_ENABLED()
  GRAPH_DEBUG("Before fuseFrozenConvAddReluImpl: ", *graph);
  SubgraphRewriter rewriter;

  // CUDNN does not support conv1d
  std::array<std::string, 2> conv_operators = {"conv2d", "conv3d"};
  std::array<std::string, 2> add_operators = {"add", "add_"};
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};

  auto conv_relu_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
      %x = aten::${conv}(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
      %res = aten::${relu}(%x)
      return (%res))");

#ifdef USE_ROCM
  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::miopen_convolution_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";
#else
  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::cudnn_convolution_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";
#endif

  auto conv_add_relu_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
      %x = aten::${conv}(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
      %y = aten::${add}(%x, %z, %alpha)
      %res = aten::${relu}(%y)
      return (%res))");

#ifdef USE_ROCM
  std::string conv_add_relu_fused = R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::miopen_convolution_add_relu(%input, %weight, %z, %alpha, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";
#else
  std::string conv_add_relu_fused = R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::cudnn_convolution_add_relu(%input, %weight, %z, %alpha, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";
#endif

  for (const auto& conv : conv_operators) {
    for (const auto& relu : relu_operators) {
      at::jit::TemplateEnv env;
      env.s("conv", conv);
      env.s("relu", relu);
      rewriter.RegisterRewritePattern(
          conv_relu_rstring.format(env), conv_relu_fused);
      for (const auto& add : add_operators) {
        env.s("add", add);
        rewriter.RegisterRewritePattern(
            conv_add_relu_rstring.format(env), conv_add_relu_fused);
      }
    }
  }

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    auto weight = toIValue(match.values_map.at(vmap.at("weight")));
    if (!weight.has_value() || !weight.value().isTensor()) {
      return false;
    }
    const at::Tensor& weight_t = weight.value().toTensor();
    if (!weight_t.device().is_cuda() || !weight_t.is_contiguous()) {
      return false;
    }

    // bias is optional
    if (vmap.find("bias") != vmap.end()) {
      auto bias = toIValue(match.values_map.at(vmap.at("bias")));
      if (bias.has_value() && bias.value().isTensor()) {
        const at::Tensor& bias_t = bias.value().toTensor();
        if (bias_t.dtype() != weight_t.dtype() || bias_t.ndimension() != 1 ||
            bias_t.size(0) != weight_t.size(0) || !bias_t.device().is_cuda()) {
          return false;
        }
      }
    }

    // z is optional
    if (vmap.find("z") != vmap.end()) {
      auto z = toIValue(match.values_map.at(vmap.at("z")));
      if (z.has_value() && z.value().isTensor()) {
        const at::Tensor& z_t = z.value().toTensor();
        if (z_t.dtype() != weight_t.dtype() ||
            z_t.size(0) != weight_t.size(0) || !z_t.is_contiguous() ||
            !z_t.device().is_cuda()) {
          return false;
        }
      }
    }
    return true;
  };

  // Convert _convolution and in-place operators for simpler replacement pattern
  // matching
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  rewriter.runOnGraph(graph, filter);
  GRAPH_DEBUG("After fuseFrozenConvAddReluImpl: ", *graph);
#endif
}

auto dummyInitializer = []() {
  getFuseFrozenConvAddReluImpl() = fuseFrozenConvAddReluImpl;
  return true;
}();

} // namespace

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

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

- `ATen/Utils.h`
- `ATen/code_template.h`
- `ATen/cuda/CUDAConfig.h`
- `torch/csrc/jit/ir/constants.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/subgraph_matcher.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h`
- `torch/csrc/jit/passes/graph_rewrite_helper.h`
- `torch/csrc/jit/passes/remove_mutation.h`
- `torch/csrc/jit/passes/subgraph_rewrite.h`


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

- **File Documentation**: `frozen_conv_add_relu_fusion_cuda.cpp_docs.md`
- **Keyword Index**: `frozen_conv_add_relu_fusion_cuda.cpp_kw.md`
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

- **File Documentation**: `frozen_conv_add_relu_fusion_cuda.cpp_docs.md_docs.md`
- **Keyword Index**: `frozen_conv_add_relu_fusion_cuda.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
