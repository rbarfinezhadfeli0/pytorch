# Keyword Index: `torch/csrc/jit/passes/shape_analysis.cpp`

## File Information

- **Original File**: [torch/csrc/jit/passes/shape_analysis.cpp](../../../../../torch/csrc/jit/passes/shape_analysis.cpp)
- **Documentation**: [`shape_analysis.cpp_docs.md`](./shape_analysis.cpp_docs.md)
- **Folder**: `torch/csrc/jit/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`ShapePropagator`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`node`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`register_formula_for`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)

### Functions

- **`DoesntRefineOutputs`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`EraseShapeInformation`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`PropagateCatShape`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`PropagateCompleteShapeOnNode`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`PropagateInputShapes`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`PropagateShapeOnNodeByRunningIt`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`PropagateTensorShapeOnNode`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`applyTypes`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`broadcastBinary`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`canPropagateShapeByRunningIt`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`collectResizeSet`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`containsTensorType`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`dependsOnMutation`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`getOrCreateUnshapedType`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`if`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`isValidArgumentForRunning`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`isValidReturnForRunning`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`mergeTypes`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`propagateTorchTensorShape`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`representativeValue`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`resizesInput`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`setUnshapedTypeIfAliasResizedSet`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`unionScalarTypes`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`unshapedTypeImpl`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`wrapDim`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)

### Includes

- **`ATen/DeviceGuard.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`ATen/ExpandUtils.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`ATen/Functions.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`ATen/core/symbol.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`ATen/ops/empty_strided.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`c10/util/Exception.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`c10/util/irange.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`exception`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`memory`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`sstream`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/autograd/variable.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/frontend/error_report.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/ir/alias_analysis.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/ir/constants.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/ir/ir.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/ir/ir_views.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/shape_analysis.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/utils/op_registry.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/runtime/exception_message.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/runtime/operator.h`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`utility`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`vector`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)

### Namespaces

- **`prim`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`torch`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)
- **`void`**: [shape_analysis.cpp_docs.md](./shape_analysis.cpp_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
