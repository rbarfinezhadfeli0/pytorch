# Keyword Index: `torch/csrc/jit/frontend/ir_emitter.cpp`

## File Information

- **Original File**: [torch/csrc/jit/frontend/ir_emitter.cpp](../../../../../torch/csrc/jit/frontend/ir_emitter.cpp)
- **Documentation**: [`ir_emitter.cpp_docs.md`](./ir_emitter.cpp_docs.md)
- **Folder**: `torch/csrc/jit/frontend`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`A`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`CompilationUnit`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`CondValue`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`DefContext`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`Environment`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`F1`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`F2`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`F3`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`FunctionResolver`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`Hash`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`LoopStatus`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`Refinement`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`RefinementSet`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`T`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`WithLoopStatus`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`and`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`citizen`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`in`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`instance`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`kwargs`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`param`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`the`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`to_ir`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`type`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`types`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`value`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`values`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`was`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)

### Functions

- **`And`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`Not`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`Or`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`canBeNone`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`checkApplyNumInputs`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`checkApplyNumInputsRange`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`checkBreakContinue`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`createTempName`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitAssert`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitAssignment`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitAugAssignment`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitAugAssignmentGeneric`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitAugAssignmentToSelectVar`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitAugAssignmentToSubscript`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitAugAssignmentToVar`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitBreak`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitClosure`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitCondExpr`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitContinue`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitDef`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitDelete`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitExprsAssign`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitFor`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitHasAttr`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitIf`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitIfElseBlocks`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitIsInstance`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitOutput`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitRaise`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitReturn`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitSelectAssign`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitShortCircuitLogical`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitSingleAssignment`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitStatements`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitSubscriptAssign`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitTupleAssign`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitValueToTensor`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitWhile`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`emitWith`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`eraseListLiterals`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`findInAnyFrame`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`findInParentFrame`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`findInThisFrame`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`findIsNoneRefinements`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`fmap`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`for`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`getAdjTupleIndex`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`getAugOp`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`getNodeKind`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`getOperatorOverload`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`getSliceInd`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`getSugaredVar`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`getTypeForSetStateArg`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`handleMaybeNoReturn`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`if`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`insertLoad`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`insertRefinements`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`insertStore`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`intersectSet`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`meaningfulName`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`pushFrame`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`refineAndSetDictTypeHintFromCandidatesVector`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`refineAndSetListTypeHintFromCandidatesVector`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`refineAndSetUnionTypeHintOrPopulateCandidatesVector`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`removeVar`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`reportSourceLocation`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`reverseComparision`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`runCleanupPasses`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`sameVar`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`setSugaredVar`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`setType`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`setVar`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`shouldDeriveSetStateType`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`throwVarNotFoundError`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`type`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`unionSet`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`validateAssignLhsExpr`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)

### Includes

- **`ATen/core/interned_strings.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`ATen/core/jit_type.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`c10/util/Exception.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`c10/util/StringUtil.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`c10/util/env.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`c10/util/hash.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`c10/util/irange.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`caffe2/serialize/versions.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`climits`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`optional`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`set`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`stack`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/api/function_impl.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/canonicalize_modified_loop.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/convert_to_ssa.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/error_report.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/ir_emitter.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/lexer.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/parser.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/schema_matching.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/script_type_parser.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/frontend/tree_views.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/ir/constants.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/ir/ir.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/annotate_warns.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/canonicalize.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_pooling.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_propagation.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/inline_forked_closures.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/inliner.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/lift_closures.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/lower_tuples.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/normalize_ops.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/passes/replacement_of_old_operators.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/runtime/graph_iterator.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/runtime/interpreter.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/runtime/operator.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/runtime/slice_indices_adjust.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch/csrc/jit/testing/hooks_for_testing.h`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)

### Namespaces

- **`namespace`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`otherwise`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)
- **`torch`**: [ir_emitter.cpp_docs.md](./ir_emitter.cpp_docs.md)


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
