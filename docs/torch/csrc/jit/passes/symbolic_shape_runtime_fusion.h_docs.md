# Documentation: `torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h`
- **Size**: 2,339 bytes (2.28 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>

namespace torch::jit {

// Takes in a TensorExprGraph of static shapes and generalizes the input shapes
// to symbolic dimensions. Dimensions of value 1 will be preserved, otherwise
// dimensions with the same value will be bucketed to the same symbolic shape.
// E.g. Tensor(5, 3), Tensor(3, 1) -> Tensor(SS(-1), SS(-2)), Tensor(SS(-2), 1)
// From there, runs symbolic shape inference on the graph, and creates a
// versioning if in the graph with prim::TensorExprDynamicGuard checking if
// the inputs at runtime match the Generalized Symbolic Shapes that are inputs
// to the TE Kernel. The computate to calculate all symbolic dimensions is
// inlined in to the if block with the TE Kernel. All Sym Dim Value* are
// appended to the end of the TE Kernel Graph/Node inputs, and the Node is
// augmented with a integer list attr `symbolic_shape_inputs` that gives the
// mapping from Value * -> Symbolic Shape int64_t value. For more lengthy IR
// examples and walkthrough look at ShapeAnalysisTest.DynamicShapesFusion in
// `test_shape_analysis` Returns True on Success, False on Failure, can fail if
// shape propagation fails to propagate # of dims or if complete shapes on
// inputs not set

TORCH_API bool GenerateGuard(
    Node* tensorexpr_graph_node,
    bool add_composed_op = false);

TORCH_API void runTensorExprDynamicGroup(const Code& code, Stack& stack);

enum class StrideInput {
  // Tensors natively store whether they are contiguous or not as a property
  // this makes it faster to query `is_contiguous` or
  // `is_contiguous(memory_format=channels_last)`
  // than looping through the sizes/strides yourself
  // For tensors with these properties, we only store one value:
  TENSOR_CONT,
  TENSOR_CONT_CHANNELS_LAST,
  // now, we describe other cases, where there is one stride enum
  // per dimension
  S_ONE, // STRIDE_ONE: packed
  S_CONT, // STRIDE_CONTIGUOUS: stride[i + 1] * sizes[i + 1]
  S_TRAN_CONT, // STRIDE_TRANSPOSED_CONTIGUOUS: stride[i-1] * sizes[i-1]
  S_AS_ARG, // STRIDE_AS_ARG: stride passed in as runtime value
};

TORCH_API std::string toString(StrideInput si);
TORCH_API StrideInput strideInputFromString(const std::string& si);

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `StrideInput`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/passes/symbolic_shape_analysis.h`


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

- **File Documentation**: `symbolic_shape_runtime_fusion.h_docs.md`
- **Keyword Index**: `symbolic_shape_runtime_fusion.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
