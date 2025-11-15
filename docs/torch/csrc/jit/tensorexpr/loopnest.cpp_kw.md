# Keyword Index: `torch/csrc/jit/tensorexpr/loopnest.cpp`

## File Information

- **Original File**: [torch/csrc/jit/tensorexpr/loopnest.cpp](../../../../../torch/csrc/jit/tensorexpr/loopnest.cpp)
- **Documentation**: [`loopnest.cpp_docs.md`](./loopnest.cpp_docs.md)
- **Folder**: `torch/csrc/jit/tensorexpr`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`CacheReplacer`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`ContainedStmtsFinder`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`FunctionInliner`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`IfThenElseReplacer`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`IndexFlattener`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`LoadOrStoreUseFinder`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`LoopComputeAtRewriter`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`RfactorStoreRewriter`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`StmtDeleter`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`VarNameSanitizer`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`Vectorizer`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`for`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`the`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)

### Functions

- **`FlattenIndexes`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`areConstantsAndSorted`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`areEqual`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`areIndicesLoopIndependent`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`computeInlineImpl`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`doesExprContainAnyVar`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`flatten`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`for`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`getIndexVarNameAtLevel`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`getNextAvailableName`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`getStoreStmtOfProducer`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`if`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`isConditionOptimizable`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`isConditionalFromCat`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`isTrivialPermutation`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`isValidIdentifierChar`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`isValidPermutation`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`mutate_loads`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`sanitizeName`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`success`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`try_vectorize`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`vectorize`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`vectorize_inputs`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)

### Includes

- **`ATen/core/functional.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`algorithm`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`c10/util/Logging.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`c10/util/irange.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`iostream`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`stdexcept`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/analysis.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/bounds_inference.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/eval.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/expr.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/ir.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/ir_cloner.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/ir_mutator.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/ir_printer.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/ir_simplifier.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/ir_verifier.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/loopnest.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/tensor.h`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`unordered_map`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`unordered_set`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`utility`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`vector`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)

### Namespaces

- **`analysis`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`bool`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)
- **`torch`**: [loopnest.cpp_docs.md](./loopnest.cpp_docs.md)


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
