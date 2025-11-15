# Documentation: `torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.cpp`
- **Size**: 2,751 bytes (2.69 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch::jit {
namespace onnx {
using namespace ::c10::onnx;
}

// For ONNX opset < 9, constant operator supports only three data types:
// float16, float, and double. Constants of other data types are exported as
// float or double and then cast back to their original data type with a cast
// node. The above transformation is done in this pass. The motivation behind
// having it as a post process pass opposed to handling in symbolic, is that
// many constant operators would have already been removed in the export before
// this step. On the other hand if cast is inserted in symbolic, subsequent node
// conversion will break if it depends on certain inputs being constant.
static void CastAllConstantToFloating(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      CastAllConstantToFloating(block);
    }

    if (node->kind() == onnx::Constant) {
      auto val = node->t(attr::value);
      at::ScalarType dtype = val.scalar_type();
      auto val_type = TensorType::create(val);
      if (dtype != at::ScalarType::Double && dtype != at::ScalarType::Float &&
          dtype != at::ScalarType::Half) {
        int to_type = 0;
        switch (val.scalar_type()) {
          case at::ScalarType::Byte:
          case at::ScalarType::Char:
          case at::ScalarType::Int:
          case at::ScalarType::Short:
          case at::ScalarType::Bool:
            to_type = ATenTypeToOnnxType(val.scalar_type());
            val = val.to(at::ScalarType::Float);
            break;

          case at::ScalarType::Long:
            to_type = ATenTypeToOnnxType(val.scalar_type());
            val = val.to(at::ScalarType::Double);
            break;

          default:
            throw std::runtime_error("Unsupported types: complex, string");
        }
        // create a cast node
        node->removeAttribute(attr::value);
        node->t_(attr::value, val);
        Node* cast_node = graph->create(onnx::Cast, 1);
        cast_node->i_(attr::to, to_type);
        cast_node->output()->setType(val_type);
        cast_node->insertAfter(node);
        // get input from cast node
        node->outputs().at(0)->replaceAllUsesWith(cast_node->outputs().at(0));
        // add input from constant to cast node
        cast_node->addInput(node->outputs().at(0));
        cast_node->copyMetadata(node);
      }
    }
  }
}

void CastAllConstantToFloating(const std::shared_ptr<Graph>& graph) {
  CastAllConstantToFloating(graph->block());
}
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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

- `torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.h`
- `torch/csrc/jit/passes/onnx/helper.h`


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

- **File Documentation**: `cast_all_constant_to_floating.cpp_docs.md`
- **Keyword Index**: `cast_all_constant_to_floating.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
