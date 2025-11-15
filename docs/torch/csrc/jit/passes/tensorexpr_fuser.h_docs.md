# Documentation: `torch/csrc/jit/passes/tensorexpr_fuser.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/tensorexpr_fuser.h`
- **Size**: 2,699 bytes (2.64 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch::jit {

// Run TensorExpressions-based fuser.
// If add_composed_op is true, creates a single operation that
// performs both the runtime check that types align
// and then the dispatch to the kernel/unoptimized graph
TORCH_API void FuseTensorExprs(
    std::shared_ptr<Graph>& graph,
    size_t min_group_size = 2,
    bool add_composed_op = false,
    bool fuse_to_dynamic_shapes = false);

TORCH_API void setTensorExprFuserEnabled(bool val);
TORCH_API bool tensorExprFuserEnabled();
TORCH_API void setTensorExprDynamicShapeFusionEnabled(bool val);
TORCH_API bool tensorExprDynamicShapeFusionEnabled();
TORCH_API bool setTexprReductionsEnabled(bool value);
TORCH_API bool texprReductionsEnabled();

TORCH_API void RemoveProfileNodesAndSpecializeTypes(
    std::shared_ptr<Graph>& graph);
TORCH_API bool hasTensorTypeSpecialization(Value* v);
TORCH_API void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph);
TORCH_API void removeTensorTypeSpecializations(Block* block);

using tensor_type_converter_t =
    c10::function_ref<TensorTypePtr(const TensorTypePtr& t)>;

// inserts a TypeCheck pattern
//
// around the guarded node that has a Subgraph attribute, this inserts a pattern
//
//   if TypeCheck(...):
//     guarded_node
//   else:
//     FallbackGraph(...)
//
// The TypeCheck includes the types of all Tensor inputs to the guarded_node,
// as processed by the type_converter, a lambda
// TensorTypePtr(const TensorTypePtr& t). This allows to erase irrelevant
// aspects of the type.
//
// The Fallback graph will have the same subgraph as the guarded node (with the
// expectation that the guarded_node's subgraph will then be optimized.
TORCH_API void insertTypeGuard(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    c10::Symbol kind);

TORCH_API bool usedOnlyInSize(Value* v);
TORCH_API Value* broadcastSizes(at::ArrayRef<Value*> sizes, AliasDb* db);

namespace tensorexpr {
TORCH_API bool isSupported(Node* node);

/// Get the modifiable custom operator set object.
///
/// For static shapes, if a custom operator has been added to the custom
/// operator set, it will be pulled into the NNC fusion group. But it doesn't
/// work with dynamic shapes unless explicitly register the shape function via
/// `torch::jit::RegisterShapeComputeGraphForSchema` for the custom operator.
///
/// @return Reference of the custom operator set
///
TORCH_API OperatorSet& getCustomOperatorSet();

} // namespace tensorexpr
} // namespace torch::jit

C10_DECLARE_bool(torch_jit_disable_cat);
C10_DECLARE_bool(torch_jit_enable_dynamic_shape_fusion);

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `tensorexpr`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/csrc/jit/ir/ir.h`
- `memory`


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

- **File Documentation**: `tensorexpr_fuser.h_docs.md`
- **Keyword Index**: `tensorexpr_fuser.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
