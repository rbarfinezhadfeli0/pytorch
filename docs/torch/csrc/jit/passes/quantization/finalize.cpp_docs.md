# Documentation: `torch/csrc/jit/passes/quantization/finalize.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/quantization/finalize.cpp`
- **Size**: 10,497 bytes (10.25 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/quantization/finalize.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/quantization/quantization_patterns.h>
#include <torch/csrc/jit/passes/quantization/register_packed_params.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#include <utility>

namespace torch::jit {

namespace {

void insertPrepackUnpackForLinear(std::shared_ptr<Graph>& graph) {
  std::vector<QuantFusionInfo> patterns_and_replacements =
      linear_prepack_unpack_patterns();

  for (const auto& entry : patterns_and_replacements) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(entry.pattern, entry.replacement);
    rewriter.runOnGraph(graph, entry.filters);
  }
}

void insertPrepackUnpackForConv(std::shared_ptr<Graph>& graph) {
  std::vector<QuantFusionInfo> patterns_and_replacements =
      conv_prepack_unpack_patterns();

  for (const auto& entry : patterns_and_replacements) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(entry.pattern, entry.replacement);
    rewriter.runOnGraph(graph, entry.filters);
  }
}

void removePackedParamInsertionAndFPWeightsSetAttr(
    std::shared_ptr<Graph>& g,
    const std::unordered_set<std::string>& packed_param_attr_names) {
  DepthFirstGraphNodeIterator it(g);
  Node* n = nullptr;
  std::vector<Node*> nodes_to_delete;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == prim::SetAttr) {
      const std::string& attr_name = n->s(attr::name);
      if (packed_param_attr_names.count(attr_name)) {
        nodes_to_delete.push_back(n);
      } else {
        Value* v = n->input(0);
        Value* self = g->inputs()[0];
        std::vector<std::string> paths = getModuleAccessPath(v, self);
        std::string path = joinPaths(paths);
        if (packed_param_attr_names.count(path)) {
          nodes_to_delete.push_back(n);
        }
      }
    }
  }
  for (auto node : nodes_to_delete) {
    node->removeAllInputs();
  }
  for (auto node : nodes_to_delete) {
    node->destroy();
  }
  ConstantPooling(g);
  EliminateDeadCode(g);
}

void removeObserverCallMethods(std::shared_ptr<Graph>& g) {
  DepthFirstGraphNodeIterator it(g);
  Node* n = nullptr;
  std::vector<Node*> nodes_to_delete;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == prim::CallMethod) {
      const std::string& attr_name = n->s(attr::name);
      if (attr_name == "calculate_qparams") {
        auto observer_node = n->input(0)->node();
        if (observer_node->kind() == prim::GetAttr &&
            observer_node->s(attr::name).find("_observer_") !=
                std::string::npos) {
          nodes_to_delete.push_back(n);
        }
      }
    }
  }
  for (auto node : nodes_to_delete) {
    node->removeAllInputs();
  }
  for (auto node : nodes_to_delete) {
    node->destroy();
  }
  EliminateDeadCode(g);
}

void keepOnlyPackedParamsGeneration(Module& m, const std::string& method_name) {
  auto g = m.get_method(method_name).graph();
  Function& function = m.get_method(method_name).function();
  const auto& schema = function.getSchema();
  auto new_schema = schema.cloneWithReturns({Argument("", NoneType::get())});
  for (size_t i = 0, output_size = g->outputs().size(); i < output_size; i++) {
    g->eraseOutput(i);
  }
  Node* none_node = g->createNone();
  g->registerOutput(none_node->output());
  none_node->insertBefore(g->return_node());
  function.setSchema(std::move(new_schema));
  EliminateDeadCode(g);
}

} // namespace

void QuantFusion(std::shared_ptr<Graph>& graph, QuantType quant_type) {
  std::vector<QuantFusionInfo> patterns;
  if (quant_type == QuantType::DYNAMIC) {
    patterns = dynamic_quant_fusion_pattern_and_replacements();
    std::vector<QuantFusionInfo> patterns_wo_dynamic_activation_quant =
        dynamic_quantized_linear_pattern_and_replacements();
    patterns.insert(
        patterns.end(),
        patterns_wo_dynamic_activation_quant.begin(),
        patterns_wo_dynamic_activation_quant.end());
  } else {
    patterns = quant_fusion_pattern_and_replacements();
  }
  for (const auto& info : patterns) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(info.pattern, info.replacement);
    rewriter.runOnGraph(graph, info.filters);
  }
}

void InsertPrepackUnpack(std::shared_ptr<Graph>& graph) {
  insertPrepackUnpackForLinear(graph);
  insertPrepackUnpackForConv(graph);
}

void InsertPrepackUnpack(Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    InsertPrepackUnpack(graph);
  }
  for (Module m : module.children()) {
    InsertPrepackUnpack(m);
  }
}

void FoldQuantizedPrepackingOps(Module& module) {
  auto filter_fn = [](const Node* n) -> bool {
    return (
        n->kind() == Symbol::fromQualString("quantized::linear_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv1d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv3d_prepack") ||
        n->kind() ==
            Symbol::fromQualString("quantized::conv_transpose1d_prepack") ||
        n->kind() ==
            Symbol::fromQualString("quantized::conv_transpose2d_prepack"));
  };
  PrePackingOpsFolder(module, filter_fn, "quantized");
}

static std::unordered_set<std::string> RegisterPrePackingParams(
    Module& module,
    const std::string& method_name) {
  auto filter_fn = [](const Node* n) -> bool {
    return (
        n->kind() == Symbol::fromQualString("quantized::linear_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv1d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv3d_prepack") ||
        n->kind() ==
            Symbol::fromQualString("quantized::conv_transpose1d_prepack") ||
        n->kind() ==
            Symbol::fromQualString("quantized::conv_transpose2d_prepack"));
  };
  return RegisterPrePackParams(module, method_name, filter_fn, "");
}

Module Finalize(
    Module& module,
    QuantType quant_type,
    const std::vector<std::string>& preserved_attrs) {
  // Tracing annotates the resulting graph with shape information. In many case,
  // user applies different input shapes to traced graph. It is on the user to
  // know it is correct to do so. The quantized module needs to be clean up and
  // To prevent the JIT optimizations from leveraging the annotated shape info,
  // clear shape information in the graph.
  for (auto func : module.type()->methods()) {
    ClearProfilingInformation(toGraphFunction(*func).graph());
  }

  auto graph = module.get_method("forward").graph();
  InsertPrepackUnpack(graph);
  GRAPH_DUMP("Before QuantFusion:", graph);
  QuantFusion(graph, quant_type);
  auto frozen = freeze_module(module, preserved_attrs);
  FoldQuantizedPrepackingOps(frozen);
  return frozen;
}

Module FinalizeOnDevicePTQ(
    Module& module,
    QuantType quant_type,
    const std::string& method_name) {
  // Tracing annotates the resulting graph with shape information. In many case,
  // user applies different input shapes to traced graph. It is on the user to
  // know it is correct to do so. The quantized module needs to be clean up and
  // To prevent the JIT optimizations from leveraging the annotated shape info,
  // clear shape information in the graph.
  for (auto func : module.type()->methods()) {
    ClearProfilingInformation(toGraphFunction(*func).graph());
  }

  const std::string kQuantizeString = "quantize_";
  const auto matched_pos = method_name.find(kQuantizeString);
  const auto end_pos = matched_pos + kQuantizeString.length();
  const std::string orig_method_name = method_name.substr(end_pos);
  TORCH_CHECK(
      matched_pos == 0,
      "Quantized ops can only be added to quantize_",
      orig_method_name,
      ". Please make sure to run quant/dequant nodes insertion step for on-device PTQ.");

  const std::string quantized_method_name = "quantized_" + orig_method_name;
  auto graph = module.get_method(method_name).graph();
  // Doing some AOT optimizations here
  // Of all CSE seems to be required otherwise in some experiments
  // serialized model is incorrect. As in it cannot be deserialized
  // Rest are included as canonical optimizations that are not for inference
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
  PeepholeOptimize(graph);
  ConstantPropagation(graph);
  UnrollConstantLoops(graph);
  ConstantPooling(graph);

  InsertPrepackUnpack(graph);
  GRAPH_DUMP("Before QuantFusion:", graph);
  QuantFusion(graph, quant_type);
  auto packed_param_attr_names = RegisterPrePackingParams(module, method_name);
  GRAPH_DUMP("After QuantFusion + packed param registration:", graph);

  // Now we have:
  // 1. Inserted quantized weights packed params
  // 2. Inserted packed params to module
  // 3. Inserted quantized op
  // The next thing we need is:
  // 1. Replicate this method in quantize_forward
  // 2. Remove SetAttr for fp weights that are reset by quantize_forward
  // 3. Remove SetAttr node which will subsequently optimize away the nodes
  //    producing packed_params
  // 4. Modify quantized_forward to remove all the nodes except for SetAttrs
  cloneMethod(module, method_name, quantized_method_name);
  // removeWeightSetAttrs(module, quantized_method_name);
  auto quantized_graph = module.get_method(quantized_method_name).graph();
  removePackedParamInsertionAndFPWeightsSetAttr(
      quantized_graph, packed_param_attr_names);
  // Removing packed params is not sufficient since that does not do DCE
  // for observer node's getatts and callmethods because callmethods have side
  // effects
  removeObserverCallMethods(quantized_graph);
  // This step removed the return output from the graph and subsequent
  // DCE removes all the ops. After that only remaining things should be
  // packed_params
  keepOnlyPackedParamsGeneration(module, method_name);
  return module;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/quantization/finalize.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/clear_profiling.h`
- `torch/csrc/jit/passes/common_subexpression_elimination.h`
- `torch/csrc/jit/passes/constant_pooling.h`
- `torch/csrc/jit/passes/constant_propagation.h`
- `torch/csrc/jit/passes/dead_code_elimination.h`
- `torch/csrc/jit/passes/freeze_module.h`
- `torch/csrc/jit/passes/loop_unrolling.h`
- `torch/csrc/jit/passes/peephole.h`
- `torch/csrc/jit/passes/prepack_folding.h`
- `torch/csrc/jit/passes/quantization/quantization_patterns.h`
- `torch/csrc/jit/passes/quantization/register_packed_params.h`
- `torch/csrc/jit/runtime/graph_iterator.h`
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

Files in the same folder (`torch/csrc/jit/passes/quantization`):

- [`quantization_type.cpp_docs.md`](./quantization_type.cpp_docs.md)
- [`insert_observers.cpp_docs.md`](./insert_observers.cpp_docs.md)
- [`insert_quant_dequant.h_docs.md`](./insert_quant_dequant.h_docs.md)
- [`register_packed_params.h_docs.md`](./register_packed_params.h_docs.md)
- [`helper.cpp_docs.md`](./helper.cpp_docs.md)
- [`finalize.h_docs.md`](./finalize.h_docs.md)
- [`insert_observers.h_docs.md`](./insert_observers.h_docs.md)
- [`fusion_passes.h_docs.md`](./fusion_passes.h_docs.md)
- [`quantization_patterns.h_docs.md`](./quantization_patterns.h_docs.md)


## Cross-References

- **File Documentation**: `finalize.cpp_docs.md`
- **Keyword Index**: `finalize.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
