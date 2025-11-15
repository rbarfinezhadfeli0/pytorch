# Documentation: `docs/torch/csrc/jit/passes/quantization/register_packed_params.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/quantization/register_packed_params.cpp_docs.md`
- **Size**: 9,255 bytes (9.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/quantization/register_packed_params.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/quantization/register_packed_params.cpp`
- **Size**: 6,478 bytes (6.33 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <stack>

#include <ATen/ATen.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/quantization/register_packed_params.h>

namespace torch::jit {

namespace {
bool isPrepackNode(Node* n) {
  return (
      n->kind() == Symbol::fromQualString("quantized::linear_prepack") ||
      n->kind() == Symbol::fromQualString("quantized::conv1d_prepack") ||
      n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
      n->kind() == Symbol::fromQualString("quantized::conv3d_prepack") ||
      n->kind() ==
          Symbol::fromQualString("quantized::conv_transpose1d_prepack") ||
      n->kind() ==
          Symbol::fromQualString("quantized::conv_transpose2d_prepack"));
}

std::pair<Value*, std::string> findFPWeight(Node* prepack_node) {
  TORCH_CHECK(isPrepackNode(prepack_node));
  Node* n = nullptr;
  n = prepack_node->input(0)->node();
  bool is_quantize_node =
      (n->kind() == Symbol::fromQualString("aten::quantize_per_tensor") ||
       n->kind() == Symbol::fromQualString("aten::quantize_per_channel"));
  TORCH_CHECK(
      is_quantize_node,
      "Input to prepack node must be output of weight quantization.");
  // First input of quantize node is FP32 weight
  n = n->input(0)->node();
  bool is_getattr_node = (n->kind() == prim::GetAttr);
  if (is_getattr_node) {
    return {n->input(0), n->s(attr::name)};
  }
  return {nullptr, "AttributeDoesNotExist"};
}
} // namespace

std::string joinPaths(const std::vector<std::string>& paths) {
  std::string path;
  for (const auto& p : paths) {
    path.append(p).append(".");
  }
  return path;
}
// Must run this pass after constant folding.
std::unordered_set<std::string> RegisterPrePackParams(
    Module& m,
    const std::string& method_name,
    const PrePackParamFilterFn& is_packed_param,
    const std::string& attr_prefix) {
  int64_t uid = 0; // int + method name gives unique identifier
  auto graph = m.get_method(method_name).graph();
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  std::string attr_name_base =
      attr_prefix + "_" + method_name + "_ondevice_ptq_packed_weight_";
  std::unordered_set<std::string> packed_param_names;

  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (is_packed_param(n)) {
        WithInsertPoint ins(n->next());
        Value* packed_param_value = n->output(0);
        TORCH_CHECK(n->outputs().size() == 1, "Prepack ops have single output");
        auto attr_name = attr_name_base + std::to_string(uid++);
        TORCH_CHECK(
            packed_param_value->uses().size() == 1,
            "Packed param must be used by exactly one op.");
        auto use = packed_param_value->uses()[0];
        while (m.hasattr(attr_name)) {
          attr_name = attr_name_base + "_" + std::to_string(uid++);
        }
        // Now register attribute for this packed param but dont set it to any
        // value. No value because we dont know what the value is at this point.
        // Only when we run on-device ptq workflow, e.g. run quantize_forward
        // method, is when the linear_prepack op will be executed and at that
        // point we will have the actual value for this attribute.
        m.register_attribute(attr_name, n->output(0)->type(), IValue());
        // In order to add the output of linear_prepack, we now have to do
        // setAttr Thus when quantize_forward is actually called the attribute
        // is appropriately set.
        Node* set_attr = graph->createSetAttr(
            graph->inputs()[0], attr_name, packed_param_value);
        set_attr->insertAfter(n);
        // Now let's add GetAttr for the same attribute.
        // Why?
        // Because eventually the method being modified will be cloned into
        // quantize_forward and quantized_forward.
        // quantize_forward will only have, for example, linear_prepack and
        // SetAttr Thus when quantize_forward is run attributes on the module
        // are set. Then in quantized_forward we will actually get
        // packed_params, via GetAttr and supply it to, for example,
        // dynamic_linear At the end quantize_forward will not have any ops like
        // dynamic_linear and quantized_forward will not have any linear_prepack
        // or SetAttr
        Value* packed_param_attr =
            graph->insertGetAttr(graph->inputs()[0], attr_name)
                ->setType(n->output(0)->type());
        // We must replace this specific usage and we cannot doe
        // replaceAllUsesWith This is because we first had to insert SetAttr
        // node. This also takes as input packed_param_value, similar to the
        // actual op. But only the use of the actual op must be replaced by
        // output of GetAttr. Input of SetAttr still must use the
        // packed_param_value
        use.user->replaceInput(use.offset, packed_param_attr);
        // Record the name of the attribute so that we can delete the SetAttr
        // for it
        packed_param_names.insert(std::move(attr_name));

        // Now make sure that original weight is reset such that the module
        // does not have weight attribute set anymore
        auto value_weight_names_pair = findFPWeight(n);
        Value* v = value_weight_names_pair.first;
        std::string weight_name = std::move(value_weight_names_pair.second);
        auto empty_tensor =
            at::empty({0}, at::TensorOptions().requires_grad(false));
        Node* none_node = graph->create(prim::Constant);
        none_node->t_(attr::value, empty_tensor);
        // none_node->output()->setType(TensorType::create(at::kFloat,
        // c10::kCPU, 1, false));
        Node* set_attr_orig_weight =
            graph->createSetAttr(v, weight_name, none_node->output());
        set_attr_orig_weight->insertAfter(packed_param_attr->node());
        none_node->insertBefore(set_attr_orig_weight);
        auto* self = v->owningGraph()->inputs()[0];
        std::vector<std::string> path = getModuleAccessPath(v, self);
        packed_param_names.emplace(joinPaths(path));
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return packed_param_names;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `std`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `stack`
- `ATen/ATen.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/passes/constant_pooling.h`
- `torch/csrc/jit/passes/constant_propagation.h`
- `torch/csrc/jit/passes/quantization/helper.h`
- `torch/csrc/jit/passes/quantization/register_packed_params.h`


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
- [`finalize.cpp_docs.md`](./finalize.cpp_docs.md)
- [`helper.cpp_docs.md`](./helper.cpp_docs.md)
- [`finalize.h_docs.md`](./finalize.h_docs.md)
- [`insert_observers.h_docs.md`](./insert_observers.h_docs.md)
- [`fusion_passes.h_docs.md`](./fusion_passes.h_docs.md)
- [`quantization_patterns.h_docs.md`](./quantization_patterns.h_docs.md)


## Cross-References

- **File Documentation**: `register_packed_params.cpp_docs.md`
- **Keyword Index**: `register_packed_params.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes/quantization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes/quantization`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/passes/quantization`):

- [`dedup_module_uses.h_kw.md_docs.md`](./dedup_module_uses.h_kw.md_docs.md)
- [`insert_observers.cpp_kw.md_docs.md`](./insert_observers.cpp_kw.md_docs.md)
- [`insert_quant_dequant.cpp_kw.md_docs.md`](./insert_quant_dequant.cpp_kw.md_docs.md)
- [`finalize.cpp_kw.md_docs.md`](./finalize.cpp_kw.md_docs.md)
- [`register_packed_params.h_kw.md_docs.md`](./register_packed_params.h_kw.md_docs.md)
- [`helper.cpp_docs.md_docs.md`](./helper.cpp_docs.md_docs.md)
- [`fusion_passes.h_kw.md_docs.md`](./fusion_passes.h_kw.md_docs.md)
- [`finalize.cpp_docs.md_docs.md`](./finalize.cpp_docs.md_docs.md)
- [`quantization_type.h_docs.md_docs.md`](./quantization_type.h_docs.md_docs.md)
- [`insert_observers.cpp_docs.md_docs.md`](./insert_observers.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `register_packed_params.cpp_docs.md_docs.md`
- **Keyword Index**: `register_packed_params.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
