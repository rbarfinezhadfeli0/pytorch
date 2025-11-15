# Documentation: `torch/csrc/jit/passes/onnx/shape_type_inference.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/shape_type_inference.h`
- **Size**: 3,955 bytes (3.86 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>

#include <utility>

namespace torch::jit {

// Merges existing_type and inferred_type.
// Returns {merged type, whether or not inferred_type was used}.
//
// The inferred type will take higher precedence, since it is produced by ONNX
// shape inference, and is more compatible with ONNX. In cases where ONNX shape
// inference fails to produce an inferred type, or produces an inferred type
// that is incomplete, refer to existing type and fill in the gap that is
// missing. Currently the following cases are supported.
//  1. existing type: Tensor[], inferred type: Tensor[]
//    For list of tensors, existing type does not store datatype nor shape for
//    inner tensor. Thus inferred type always contain more information, and is
//    returned.
//  2. existing type: Tensor, inferred type: Tensor
//    Fill in missing info (shape, data type) for inferred type from existing
//    type.
//  3. existing type: Scalar[], inferred type: Tensor
//    ONNX represents list of scalars by 1-d Tensor. Return inferred type since
//    it is more compatible with ONNX.
std::pair<TypePtr, bool> MergeInferredType(
    const TypePtr& existing_type,
    const TypePtr& inferred_type);

void MergeInferredTypeAndSetMap(
    Value* dest_v,
    const TypePtr& existing_type,
    const TypePtr& inferred_type);

// Update graph input types with dynamic axes info.
// Axes that are marked as dynamic will be assigned as dynamic ShapeSymbol.
// Note it is possible for multiple axes to share the same ShapeSymbol,
// if they are defined as such in dynamic_axes.
TORCH_API void ONNXSetDynamicInputShape(
    std::shared_ptr<Graph>& graph,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    const std::vector<std::string>& input_names);

// Update graph output with types of output Tensors.
// If onnx_shape_inference is true, types of output Tensors will be compared and
// merged with inferred types. It is possible that inferred types contain
// dynamic axes, hence it takes precedence over types of output Tensors.
TORCH_API void ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    at::ArrayRef<at::Tensor> outputs,
    const python::IODescriptor& desc,
    bool onnx_shape_inference,
    bool is_script,
    int opset_version);

// Replace None in output with Optional node (opset > 15) if it's
// script model. This helps align the output format in ONNX internal tests
// when comparing pytorch results with ONNX results, as they have different
// process for None in output.
void ReplaceGraphOutputNoneWithOptional(
    std::shared_ptr<Graph>& graph,
    size_t outputs_index);
Node* ONNXOptionalNodeForNone(std::shared_ptr<Graph>& graph);

// Utilize ONNX Shape Inference for node.
// The node must have ONNX namespace, and is valid ONNX node according to spec.
// On successful ONNX shape inference runs, the function updates output types of
// n with inferred shape and type. Otherwise n is unchanged.
TORCH_API void ONNXShapeTypeInference(
    Node* n,
    const ParamMap& params_dict,
    int opset_version);

// Utilize ONNX Shape Inference for graph.
// Internally calls ONNXShapeTypeInference for each node, to achieve more
// coverage that skips only individual nodes if illegal, instead of skipping for
// the entire graph.
TORCH_API void ONNXShapeTypeInference(
    std::shared_ptr<Graph>& g,
    const ParamMap& params_dict,
    int opset_version);

bool AllGraphInputsStatic(const Graph* g);
std::pair<bool, bool> AreInputsReliableOrStatic(Node* n);
void UpdateReliable(
    torch::jit::Value* output,
    const std::pair<bool, bool>& input_reliable,
    bool no_type_warning = false);

void UpdateReliable(torch::jit::Node* n);
void UpdateShapeConstantIfReliable(torch::jit::Value* output);

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/passes/onnx/helper.h`
- `torch/csrc/jit/python/python_arg_flatten.h`
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

- **File Documentation**: `shape_type_inference.h_docs.md`
- **Keyword Index**: `shape_type_inference.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
