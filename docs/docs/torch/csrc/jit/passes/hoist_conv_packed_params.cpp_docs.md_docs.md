# Documentation: `docs/torch/csrc/jit/passes/hoist_conv_packed_params.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/hoist_conv_packed_params.cpp_docs.md`
- **Size**: 7,706 bytes (7.53 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/hoist_conv_packed_params.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/hoist_conv_packed_params.cpp`
- **Size**: 4,833 bytes (4.72 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <stack>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/hoist_conv_packed_params.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

namespace torch::jit {

// Hoists packed params from a conv module to the parent module.
// The benefit is that after this hoisting, the conv module
// no longer holds anything and can be deleted, reducing model
// size.
//
// Before (easy case):
//
// %1 = prim::GetAttr[name="conv1"][%self]
// %2 = prim::GetAttr[name="_packed_params][%1]
//
// After (easy case):
//
// %2 = prim::GetAttr[name="{prefix}.conv1._packed_params"][%self]
//
// Before (generic case):
//
// %1 = prim::GetAttr[name="name1"][%self]
// %2 = prim::GetAttr[name="name2"][%1]
// ...
// %n = prim::GetAttr[name="_packed_params][%n-1]
//
// After (generic case):
//
// %n =
// prim::GetAttr[name="{prefix}.name1{...}.name(n-1)._packed_params"][%self]
//
static void hoistConvPackedParams(
    Module& rootModule,
    Node* getConvPackedParamsNode,
    const std::string& prefix,
    int& nameUniqueCounter) {
  auto method = rootModule.get_method("forward");
  auto graph = method.graph();
  Value* rootModuleAsValue = graph->inputs()[0];

  // get a path from root module to conv module
  Value* convModuleAsValue = getConvPackedParamsNode->inputs()[0];
  std::vector<std::string> rootToConvPath =
      getModuleAccessPath(convModuleAsValue, rootModuleAsValue);

  // get a module object representing the conv
  Module convModule = findChildModule(rootModule, rootToConvPath);

  // get the packed params value
  c10::IValue packedParams = convModule.attr("_packed_params");

  // create the new name

  std::string suffix;
  for (const auto& attrName : rootToConvPath) {
    suffix += attrName + ".";
  }
  std::string newNameBase = prefix + "." + suffix + "_packed_params";
  nameUniqueCounter++;
  std::string newName = newNameBase + "." + std::to_string(nameUniqueCounter);
  while (rootModule.hasattr(newName)) {
    nameUniqueCounter++;
    newName = newNameBase + "." + std::to_string(nameUniqueCounter);
  }

  // copy the packed params
  rootModule.register_attribute(newName, packedParams.type(), packedParams);

  // change target module to rootModule
  getConvPackedParamsNode->replaceInput(0, rootModuleAsValue);

  // change attribute name to new name
  getConvPackedParamsNode->s_(Symbol::attr("name"), newName);
}

void HoistConvPackedParams(script::Module& m) {
  auto method = m.get_method("forward");
  auto graph = method.graph();

  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  std::string attr_name_base = "_jit_pass_hoist_conv_packed_params";
  // counter to ensure new attribute names are unique
  int nameUniqueCounter = 0;

  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();

    for (Node* n : b->nodes()) {
      // make sure this node is fetching {foo}.{_packed_params}
      bool isGetPackedParamsNode =
          n->kind() == prim::GetAttr && n->s(attr::name) == "_packed_params";
      if (isGetPackedParamsNode) {
        // make sure the foo in {foo}.{_packed_params} is a quantized conv
        std::optional<std::string> moduleName = getModuleName(n->inputs()[0]);
        bool moduleNameIsQuantizedConv = moduleName.has_value() &&
            (moduleName.value() ==
                 "__torch__.torch.ao.nn.quantized.modules.conv.Conv1d" ||
             moduleName.value() ==
                 "__torch__.torch.ao.nn.quantized.modules.conv.Conv2d" ||
             moduleName.value() ==
                 "__torch__.torch.ao.nn.quantized.modules.conv.Conv3d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU1d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU3d" ||
             // BC Stuff
             moduleName.value() ==
                 "__torch__.torch.nn.quantized.modules.conv.Conv1d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.quantized.modules.conv.Conv2d" ||
             moduleName.value() ==
                 "__torch__.torch.nn.quantized.modules.conv.Conv3d");

        if (moduleNameIsQuantizedConv) {
          GRAPH_UPDATE("Hoisting ", *n, " to root module.");
          hoistConvPackedParams(m, n, attr_name_base, nameUniqueCounter);
        }
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }

    } // for

  } // while
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
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/constant_pooling.h`
- `torch/csrc/jit/passes/constant_propagation.h`
- `torch/csrc/jit/passes/hoist_conv_packed_params.h`
- `torch/csrc/jit/passes/quantization/helper.h`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `hoist_conv_packed_params.cpp_docs.md`
- **Keyword Index**: `hoist_conv_packed_params.cpp_kw.md`
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

- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `hoist_conv_packed_params.cpp_docs.md_docs.md`
- **Keyword Index**: `hoist_conv_packed_params.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
