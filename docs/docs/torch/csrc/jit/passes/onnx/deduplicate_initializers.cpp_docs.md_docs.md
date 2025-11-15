# Documentation: `docs/torch/csrc/jit/passes/onnx/deduplicate_initializers.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/onnx/deduplicate_initializers.cpp_docs.md`
- **Size**: 6,153 bytes (6.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/onnx/deduplicate_initializers.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/deduplicate_initializers.cpp`
- **Size**: 3,378 bytes (3.30 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/deduplicate_initializers.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <c10/util/irange.h>

namespace torch::jit {

namespace onnx {
using namespace ::c10::onnx;
}

static void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,
    ValueToParamPairMap& valsToParamsMap,
    bool (*comp)(at::Tensor&, at::Tensor&)) {
  auto is_same_tensor_as = [&valsToParamsMap, comp](Value* v1) {
    return [&valsToParamsMap, v1, comp](Value* v2) {
      if ((valsToParamsMap.find(v1) == valsToParamsMap.end()) ||
          (valsToParamsMap.find(v2) == valsToParamsMap.end())) {
        return false;
      }
      auto iv1 = valsToParamsMap.find(v1)->second.second;
      auto iv2 = valsToParamsMap.find(v2)->second.second;
      if (!iv1.isTensor() || !iv2.isTensor()) {
        return false;
      }
      auto t1 = iv1.toTensor();
      auto t2 = iv2.toTensor();
      return comp(t1, t2);
    };
  };
  std::vector<Value*> uniqueVals;
  std::vector<size_t> inputsIndicesToRemove;
  auto b = g->block();

  for (auto i : c10::irange(b->inputs().size())) {
    auto v = g->inputs().at(i);
    if (valsToParamsMap.find(v) == valsToParamsMap.end()) {
      // Skip model inputs
      continue;
    }
    auto it = std::find_if(
        uniqueVals.begin(), uniqueVals.end(), is_same_tensor_as(v));
    if (it == uniqueVals.end()) {
      uniqueVals.emplace_back(v);
    } else {
      inputsIndicesToRemove.emplace_back(i);
      auto id_node = g->create(onnx::Identity);
      id_node->insertAfter(g->block()->param_node());
      id_node->addInput(*it);
      id_node->output()->copyMetadata(v);
      id_node->copyMetadata(g->block()->param_node());
      v->replaceAllUsesWith(id_node->output());
    }
  }
  for (auto it = inputsIndicesToRemove.rbegin();
       it != inputsIndicesToRemove.rend();
       ++it) {
    valsToParamsMap.erase(g->inputs().at(*it));
    g->eraseInput(*it);
  }
}

static bool DeduplicateInitializersByDataPtr(at::Tensor& t1, at::Tensor& t2) {
  return t1.sizes().equals(t2.sizes()) && t1.strides().equals(t2.strides()) &&
      (t1.has_storage() && t2.has_storage() && t1.data_ptr() == t2.data_ptr());
}

static bool DeduplicateInitializersByValue(at::Tensor& t1, at::Tensor& t2) {
  if (t1.dtype() != t2.dtype() || !t1.sizes().equals(t2.sizes()) ||
      !t1.strides().equals(t2.strides())) {
    return false;
  }

  if (t1.device() != t2.device()) {
    return t1.to("cpu").equal(t2.to("cpu"));
  }

  return t1.equal(t2);
}

void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,
    std::map<std::string, IValue>& paramsDict,
    bool is_train) {
  auto valsToParamsMap = buildValueToParamsMap(g->block(), paramsDict);
  // ONNX spec does not support parameters with shared memory.
  // This pass de-duplicate those parameters. Training is not affected.
  DeduplicateInitializers(g, valsToParamsMap, DeduplicateInitializersByDataPtr);
  if (!is_train) {
    // More aggressive parameters de-duplication based on tensor values.
    // Producing more compact model for inference.
    // For training, this pass is disabled,
    // because parameters may be updated differently.
    DeduplicateInitializers(g, valsToParamsMap, DeduplicateInitializersByValue);
  }
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `onnx`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/onnx/deduplicate_initializers.h`
- `torch/csrc/jit/passes/onnx/helper.h`
- `c10/util/irange.h`


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
- [`list_model_parameters.cpp_docs.md`](./list_model_parameters.cpp_docs.md)
- [`preprocess_for_onnx.h_docs.md`](./preprocess_for_onnx.h_docs.md)
- [`remove_inplace_ops_for_onnx.h_docs.md`](./remove_inplace_ops_for_onnx.h_docs.md)
- [`constant_fold.cpp_docs.md`](./constant_fold.cpp_docs.md)
- [`eliminate_unused_items.cpp_docs.md`](./eliminate_unused_items.cpp_docs.md)
- [`cast_all_constant_to_floating.h_docs.md`](./cast_all_constant_to_floating.h_docs.md)
- [`list_model_parameters.h_docs.md`](./list_model_parameters.h_docs.md)
- [`shape_type_inference.cpp_docs.md`](./shape_type_inference.cpp_docs.md)
- [`constant_map.cpp_docs.md`](./constant_map.cpp_docs.md)


## Cross-References

- **File Documentation**: `deduplicate_initializers.cpp_docs.md`
- **Keyword Index**: `deduplicate_initializers.cpp_kw.md`
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

- **File Documentation**: `deduplicate_initializers.cpp_docs.md_docs.md`
- **Keyword Index**: `deduplicate_initializers.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
