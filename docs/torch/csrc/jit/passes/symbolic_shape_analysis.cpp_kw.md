# Keyword Index: `torch/csrc/jit/passes/symbolic_shape_analysis.cpp`

## File Information

- **Original File**: [torch/csrc/jit/passes/symbolic_shape_analysis.cpp](../../../../../torch/csrc/jit/passes/symbolic_shape_analysis.cpp)
- **Documentation**: [`symbolic_shape_analysis.cpp_docs.md`](./symbolic_shape_analysis.cpp_docs.md)
- **Folder**: `torch/csrc/jit/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`ShapeArg`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`ShapeArguments`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`SymbolicShapeGraphAnalyzer`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`SymbolicShapeOpAnalyzer`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`or`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)

### Functions

- **`PropagateShapesOnBlock`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`PropagateShapesOnGraph`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`applyOutputShapeToGraph`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`at`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`combine_bounds`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`extractListShape`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`has_dim`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`if`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`isListOfInts`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`isListOfListOfInts`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`isListOfTensors`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`joinPartialEvaluatedShapeGraphToLargeShapeGraph`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`len`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`mergeSymbolicShapeSets`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`refineInputUnionTypes`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`registerStitchedComputeOutput`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`replaceWithIValue`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`setSymbolicShapeAnalysisTestMode`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`shapeGraphCleanupPasses`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`substituteConstantInputs`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`substituteSymbolicProperties`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`symbolicShapeAnalysisTestModeEnabled`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`tensorShapeArg`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`unknownInteger`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`updateGraphWithSymbolicShapeEqualities`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)

### Includes

- **`ATen/core/symbol.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`algorithm`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`c10/util/Exception.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`c10/util/irange.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`memory`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/ir/alias_analysis.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/ir/constants.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/ir/ir.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/ir/ir_views.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/common_subexpression_elimination.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_pooling.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_propagation.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/integer_value_refinement.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/loop_unrolling.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/lower_tuples.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/peephole.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/peephole_list_idioms.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/peephole_non_tensor.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/remove_mutation.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/shape_analysis.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/symbolic_shape_analysis.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/symbolic_shape_cache.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/passes/tensorexpr_fuser.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/runtime/exception_message.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`torch/csrc/jit/runtime/symbolic_shape_registry.h`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`unordered_map`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`utility`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`vector`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)

### Namespaces

- **`torch`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)
- **`void`**: [symbolic_shape_analysis.cpp_docs.md](./symbolic_shape_analysis.cpp_docs.md)


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
