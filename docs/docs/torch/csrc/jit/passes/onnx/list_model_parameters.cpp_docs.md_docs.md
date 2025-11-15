# Documentation: `docs/torch/csrc/jit/passes/onnx/list_model_parameters.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/onnx/list_model_parameters.cpp_docs.md`
- **Size**: 9,576 bytes (9.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/onnx/list_model_parameters.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/list_model_parameters.cpp`
- **Size**: 6,793 bytes (6.63 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/list_model_parameters.h>

namespace torch::jit {

namespace onnx {
using namespace ::c10::onnx;
}

// findSubModuleAttr function chases getAttr chains backwards to locate the
// submodules. For example: module M {
//   attributes {
//     A = <SubModule at ...>
//   }
//   ...
//   %A = prim::GetAttr[name="A"](%self)
//   ...
//   %B = prim::GetAttr[name="B"](%A)
//   ...
//   %weight = prim::GetAttr[name="scale"](%B)
//   ...
static std::deque<std::string> findSubModuleAttr(
    Value* input,
    std::string& name,
    Module& attrModule,
    std::shared_ptr<Graph>& graph) {
  Node* node = input->node();
  std::deque<std::string> moduleNames;

  // Loop starts from inner submodule and follows the chain until reaches the
  // top module.

  while (node->outputs().at(0)->type() != graph->inputs().at(0)->type()) {
    if (node->kind() == prim::GetAttr) {
      moduleNames.push_front(node->s(attr::name));
      node = node->inputs()[0]->node();
    } else {
      return moduleNames;
    }
  }
  // Assign the inner module to attrModule.
  for (auto& moduleName : moduleNames) {
    attrModule = attrModule.attr(moduleName).toModule();
  }
  return moduleNames;
}

static Value* addParamAsArgument(
    Function* function,
    std::string& name,
    IValue& attr) {
  auto schema = function->getSchema();
  auto args = schema.arguments();
  args.emplace_back(name, nullptr, std::nullopt, attr);
  auto new_schema = FunctionSchema(
      schema.name(),
      schema.overload_name(),
      args,
      schema.returns(),
      schema.is_vararg(),
      schema.is_varret());
  function->setSchema(new_schema);
  return toGraphFunction(*function).graph()->addInput(name)->setType(
      attr.type());
}

static std::vector<IValue> getParamAttributes(
    Block* block,
    std::shared_ptr<Graph>& graph,
    const Module& module_,
    Function* function_,
    std::unordered_map<std::string, Value*>& attrValues) {
  auto isEval = !module_.hasattr("training") || !module_.is_training();

  Node* m = *block->nodes().begin();
  WithInsertPoint guard(m);

  std::vector<IValue> parameterIValues = {};
  std::unordered_set<Node*> nodesToDestroy;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    if (n->kind() == prim::GetAttr || n->kind() == prim::SetAttr) {
      if (n->kind() == prim::GetAttr) {
        for (auto use : n->output()->uses()) {
          if (use.user->kind() == prim::PythonOp)
            throw ErrorReport(n->sourceRange())
                << "Couldn't export Python method.";
        }
      }

      auto name = n->s(attr::name);
      auto attrModule = module_;
      auto input = n->inputs()[0];

      auto moduleNames = findSubModuleAttr(input, name, attrModule, graph);
      if (!attrModule.hasattr(name))
        continue;
      auto attr = attrModule.attr(name);
      Value* paramConst = nullptr;

      std::string fullName;
      for (auto& name : moduleNames) {
        fullName += name + '.';
      }
      fullName += name;

      auto type = attrModule.type();
      auto slot = *type->findAttributeSlot(name);

      // Add model_parameters and model_buffers as model inputs. Order is
      // preserved based on the appearance in the graph.
      if (type->is_parameter(slot) || type->is_buffer(slot) ||
          (attr.isObject() && !attr.toObjectRef().type()->is_module()) ||
          attr.isBool()) {
        if (attrValues.find(fullName) == attrValues.end() &&
            attr.isTensor()) { // TODO: Handle float/int
          TORCH_INTERNAL_ASSERT(attr.isTensor());
          auto tensor_ = attr.toTensor();
          if (isEval && tensor_.requires_grad()) {
            tensor_ = tensor_.detach();
            tensor_.set_requires_grad(false);
            attr = IValue(tensor_);
          }
          parameterIValues.emplace_back(attr.toTensor());
          paramConst = addParamAsArgument(function_, fullName, attr);
          attrValues.insert({fullName, paramConst});
        } else if (attr.isObject() && !attr.toObjectRef().type()->is_module()) {
          // Only below registered torch classes are supported.
          try {
            parameterIValues.emplace_back(
                script::Object(attr.toObject()).run_method("__getstate__"));
            paramConst = addParamAsArgument(function_, fullName, attr);
            attrValues.insert({fullName, paramConst});
          } catch (const std::exception&) {
            throw ErrorReport(n->sourceRange())
                << "Unknown type " << attr.type()->repr_str()
                << " encountered in handling model params."
                << " This class type does not extend __getstate__ method.";
          }
        } else if (attr.isNone() || (attr.isBool() && name == "training")) {
          // This attr is constant for ONNX.
          auto attrVal = tryInsertConstant(*graph, attr);
          n->output()->replaceAllUsesWith(*attrVal);
          nodesToDestroy.emplace(n);
        }
      }
    }

    for (Block* sub_block : n->blocks()) {
      auto nextParameterIValues =
          getParamAttributes(sub_block, graph, module_, function_, attrValues);
      parameterIValues.insert(
          std::end(parameterIValues),
          std::begin(nextParameterIValues),
          std::end(nextParameterIValues));
    }
  }
  for (auto n : nodesToDestroy) {
    n->destroy();
  }
  return parameterIValues;
}

static void insertMainModuleAsConstant(const std::shared_ptr<Graph>& graph) {
  auto* constNode = graph->create(prim::CreateObject);
  constNode->output()->setType(graph->inputs().at(0)->type());
  auto it = graph->nodes().begin();
  constNode->insertBefore(*it);
  graph->inputs().at(0)->replaceAllUsesWith(constNode->output());
  graph->eraseInput(0);
}

std::pair<Module, std::vector<IValue>> list_module_parameters(
    const Module& module) {
  Module moduleClone = module.clone(true);
  Method method = moduleClone.get_method("forward");
  auto function = &method.function();
  auto graph = toGraphFunction(*function).graph();
  // A map of names and values of referenced attributes, to avoid duplicates.
  std::unordered_map<std::string, Value*> attrValues = {};

  GRAPH_DEBUG("Fetch attributes for function: " + function->name());
  std::vector<IValue> parameterIValues = getParamAttributes(
      graph->block(), graph, moduleClone, function, attrValues);
  insertMainModuleAsConstant(graph);
  GRAPH_DEBUG("Listed parameters as inputs: ", *graph);

  return std::make_pair(moduleClone, parameterIValues);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `onnx`, `torch`

**Classes/Structs**: `type`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/frontend/error_report.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/dead_code_elimination.h`
- `torch/csrc/jit/passes/onnx/helper.h`
- `torch/csrc/jit/passes/onnx/list_model_parameters.h`


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

Files in the same folder (`torch/csrc/jit/passes/onnx`):

- [`remove_inplace_ops_for_onnx.cpp_docs.md`](./remove_inplace_ops_for_onnx.cpp_docs.md)
- [`preprocess_for_onnx.h_docs.md`](./preprocess_for_onnx.h_docs.md)
- [`remove_inplace_ops_for_onnx.h_docs.md`](./remove_inplace_ops_for_onnx.h_docs.md)
- [`constant_fold.cpp_docs.md`](./constant_fold.cpp_docs.md)
- [`eliminate_unused_items.cpp_docs.md`](./eliminate_unused_items.cpp_docs.md)
- [`cast_all_constant_to_floating.h_docs.md`](./cast_all_constant_to_floating.h_docs.md)
- [`list_model_parameters.h_docs.md`](./list_model_parameters.h_docs.md)
- [`shape_type_inference.cpp_docs.md`](./shape_type_inference.cpp_docs.md)
- [`constant_map.cpp_docs.md`](./constant_map.cpp_docs.md)


## Cross-References

- **File Documentation**: `list_model_parameters.cpp_docs.md`
- **Keyword Index**: `list_model_parameters.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes/onnx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/passes/onnx`):

- [`constant_map.cpp_kw.md_docs.md`](./constant_map.cpp_kw.md_docs.md)
- [`deduplicate_initializers.h_docs.md_docs.md`](./deduplicate_initializers.h_docs.md_docs.md)
- [`shape_type_inference.h_docs.md_docs.md`](./shape_type_inference.h_docs.md_docs.md)
- [`function_substitution.h_kw.md_docs.md`](./function_substitution.h_kw.md_docs.md)
- [`eliminate_unused_items.h_kw.md_docs.md`](./eliminate_unused_items.h_kw.md_docs.md)
- [`prepare_division_for_onnx.cpp_docs.md_docs.md`](./prepare_division_for_onnx.cpp_docs.md_docs.md)
- [`fixup_onnx_controlflow.cpp_kw.md_docs.md`](./fixup_onnx_controlflow.cpp_kw.md_docs.md)
- [`constant_fold.cpp_docs.md_docs.md`](./constant_fold.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`onnx_log.cpp_kw.md_docs.md`](./onnx_log.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `list_model_parameters.cpp_docs.md_docs.md`
- **Keyword Index**: `list_model_parameters.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
