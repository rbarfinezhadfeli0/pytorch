# Keyword Index: `torch/csrc/jit/runtime/static/passes.cpp`

## File Information

- **Original File**: [torch/csrc/jit/runtime/static/passes.cpp](../../../../../../torch/csrc/jit/runtime/static/passes.cpp)
- **Documentation**: [`passes.cpp_docs.md`](./passes.cpp_docs.md)
- **Folder**: `torch/csrc/jit/runtime/static`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`CastedBatchOneHotLengths`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ClipRangesGather`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ClipRangesGatherRangesLengthsToOffsets`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ClipRangesGatherRangesSigridHash`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ClipRangesGatherRangesX2SigridHashPrecompute`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ClipRangesGatherSigridHash`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ClipRangesToGatherToOffsets`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ClipRangesToGatherToOffsetsV2`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ConcatAddMulReplaceNaNClip`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ConcatBatchMatMulBatchGather`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`CreateOwnedRefsForSpecialValues`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`CreateOwnedRefsForSpecialValuesHelper`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`EliminateExtraPermuteOps`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`EliminateNoOpSlice`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`EliminateTrivialEquallySplit`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ForceNonEmptyOutputs`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ForceNonEmptyOutputsHelper`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`FuseClampNaNToNum`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`FuseInferenceOpsForSparseNN`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`FuseListUnpack`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`FuseSignLog1P`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`FuseTupleUnpackBlock`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`PrecomputeMultiplierShiftForSigridHash`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`PrepackWeights`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`RemoveImmutableInputDictLookups`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`RemoveUnnecessaryEmbeddingBagOutputs`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`RemoveUnnecessaryOutputs`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ReplacePermuteWithCopy`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ReplaceWithCopy`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ReplaceWithMaybeCopy`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`SplitOutPrecomputeOpsForSparseNN`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`ToLengthsToOffsets`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`UseInPlaceGetRealInputsFromOptionalInputsV2`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`UseSplitAndSqueeze`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`UseVariadicGroupedAccessor`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`UseVariadicTupleUnpack`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`eliminatePermuteOpsSoftmaxPattern`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`eliminatePermuteOpsSumPattern`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`forwardHasOp`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`graphHasOp`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`inputIsConstantInt`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`inputIsConstantList`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`isNoOpSlice`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`shouldNotFuseListUnpackSpecialCase`**: [passes.cpp_docs.md](./passes.cpp_docs.md)

### Includes

- **`torch/csrc/jit/ir/alias_analysis.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch/csrc/jit/ir/subgraph_matcher.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_pooling.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_propagation.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch/csrc/jit/passes/subgraph_rewrite.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch/csrc/jit/passes/variadic_ops.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch/csrc/jit/runtime/graph_iterator.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch/csrc/jit/runtime/static/ops.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch/csrc/jit/runtime/static/passes.h`**: [passes.cpp_docs.md](./passes.cpp_docs.md)

### Namespaces

- **`jit`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`torch`**: [passes.cpp_docs.md](./passes.cpp_docs.md)
- **`void`**: [passes.cpp_docs.md](./passes.cpp_docs.md)


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
