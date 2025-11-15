# Keyword Index: `torch/csrc/jit/passes/tensorexpr_fuser.cpp`

## File Information

- **Original File**: [torch/csrc/jit/passes/tensorexpr_fuser.cpp](../../../../../torch/csrc/jit/passes/tensorexpr_fuser.cpp)
- **Documentation**: [`tensorexpr_fuser.cpp_docs.md`](./tensorexpr_fuser.cpp_docs.md)
- **Folder**: `torch/csrc/jit/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`REQ`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`TensorExprFuser`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`instead`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`node`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`nodes`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)

### Functions

- **`FuseTensorExprs`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`RemoveProfileNodesAndSpecializeTypes`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`RemoveTensorTypeSpecializations`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`allShapesAreKnown`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`blockSize`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`canFuseOnDevice`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`canHandle`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`canMerge`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`createFusionGroups`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`createTensorExprOp`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`debugDumpFusionGroup`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`for`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`generalizeFusionGroups`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`hasConv`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`hasTensorTypeSpecialization`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`has_unsupported_pin_memory`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`inlineIfTooSmall`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`inlineSmallFusionGroups`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`insertTypeGuard`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`isFusableOnDevice`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`isSupported`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`isSupportedForBlock`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`parseTENotFuseOption`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`prepareFusionGroupAndGuardOutputs`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`removeOutputsUsedOnlyInSize`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`removeProfileNodesAndSpecializeTypes`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`removeTensorTypeSpecialization`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`removeTensorTypeSpecializations`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`run`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`setTensorExprDynamicShapeFusionEnabled`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`setTensorExprFuserEnabled`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`setTexprReductionsEnabled`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`shapeIsKnown`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`sortReverseTopological`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`tensorExprDynamicShapeFusionEnabled`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`tensorExprFuserEnabled`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`texprReductionsEnabled`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`typesAreSupported`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`unexecutedEagerOp`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`usedOnlyInSize`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)

### Includes

- **`ATen/core/interned_strings.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`ATen/core/symbol.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`ATen/record_function.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`c10/util/FunctionRef.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`c10/util/irange.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/codegen/cuda/interface.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/codegen/fuser/interface.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/ir/alias_analysis.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/jit_opt_limit.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/common_subexpression_elimination.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_pooling.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/pass_manager.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/remove_redundant_profiles.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/tensorexpr_fuser.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/utils/subgraph_utils.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/runtime/custom_operator.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/runtime/graph_executor.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/runtime/operator_options.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/runtime/symbolic_shape_registry.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/runtime/symbolic_shape_registry_util.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/kernel.h`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`utility`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)

### Namespaces

- **`class`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`tensorexpr`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)
- **`torch`**: [tensorexpr_fuser.cpp_docs.md](./tensorexpr_fuser.cpp_docs.md)


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
