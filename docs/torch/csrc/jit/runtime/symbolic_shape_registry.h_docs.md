# Documentation: `torch/csrc/jit/runtime/symbolic_shape_registry.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/symbolic_shape_registry.h`
- **Size**: 2,802 bytes (2.74 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

/*
ADDING A NEW SHAPE GRAPH:
- For one node schema, there is one corresponding registered shape compute
graph. The schema of the graph should be the same except for Tensor arguments.
For every Tensor input in operator schema, there should be a List[int]
corresponding to that Tensor's shape. For example: "aten::linear(Tensor input,
Tensor weight, Tensor? bias=None) -> Tensor" ==> def linear(input: List[int],
weight: List[int], bias: Optional[List[int]])

Additionally, arguments which are unused at the end of the schema may be left
off. This allows sharing a single graph for multiple function schemas, such as
unary operators with different trailing arguments that do not affect the output
shape.

The shape graph should return a new, unaliased List[int] (or tuple of lists for
multiple returns) and should not modify any input lists. This allows the shape
graphs to be composed and executed.

The shape analysis (particularly for non-complete, or symbolic shapes) works by
partially evaluating the JIT IR. It may be possible for a Graph to be registered
that we cannot currently partially evaluate. If this happens, please file an
issue. There are lints registered to avoid particular known patterns (continue
or break or early return in a loop). Those may be improved in the future, please
file an issue if necessary.

To debug (and write initially) the recommended flow is to define these functions
in python and iterate there. Functions should be added to
torch/jit/_shape_functions.

To test operators, the preferred flow is through OpInfos, with
`assert_jit_shape_analysis=True`. If this is not feasible, you can look at tests
in `test_symbolic_shape_analysis.py` such as `test_adaptive_avg_pool2d`.

Operators which take in a list of tensors, such as concat, are not yet
supported. Concat has been special cased and could be generalized as needed.
Please file an issue.
*/

struct BoundedShapeGraphs {
  std::shared_ptr<Graph> lower_bound;
  std::shared_ptr<Graph> upper_bound;
};

TORCH_API void RegisterShapeComputeGraphForSchema(
    const FunctionSchema& schema,
    const std::shared_ptr<Graph>& g);

TORCH_API std::optional<std::shared_ptr<Graph>> shapeComputeGraphForSchema(
    const FunctionSchema& schema);

TORCH_API std::optional<BoundedShapeGraphs> boundedGraphsForSchema(
    const FunctionSchema& schema);

TORCH_API std::vector<const FunctionSchema*> RegisteredShapeComputeSchemas();

TORCH_API void LintShapeComputeGraph(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph);

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `BoundedShapeGraphs`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/csrc/jit/ir/ir.h`


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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `symbolic_shape_registry.h_docs.md`
- **Keyword Index**: `symbolic_shape_registry.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
