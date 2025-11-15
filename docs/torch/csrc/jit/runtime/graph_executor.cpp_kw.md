# Keyword Index: `torch/csrc/jit/runtime/graph_executor.cpp`

## File Information

- **Original File**: [torch/csrc/jit/runtime/graph_executor.cpp](../../../../../torch/csrc/jit/runtime/graph_executor.cpp)
- **Documentation**: [`graph_executor.cpp_docs.md`](./graph_executor.cpp_docs.md)
- **Folder**: `torch/csrc/jit/runtime`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`CaptureList`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`DifferentiableGraphBackward`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`DifferentiableGraphOp`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`ExecutionPlan`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`Frame`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`GraphExecutor`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`GraphExecutorImpl`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`UnpackInstructions`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)

### Functions

- **`IsNewExecutorEnabled`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`addInputIValue`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`addInputVariable`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`addOutputForIValue`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`addOutputForTensor`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`aliasAnalysisInternalSpecialCase`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`capture`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`captureInputs`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`captureOutputs`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`captureTensor`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`compileSpec`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`debugSetAutodiffSubgraphInlining`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`debugSetFusionGroupInlining`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`detach`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`detachVariables`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`for`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`getAutodiffSubgraphInlining`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`getFusionGroupInlining`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`getGradient`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`mayIntroduceGradient`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`needsGradient`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`packGradient`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`produceOutput`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`pushNone`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`pushTensor`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`pushTensorList`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`release_variables`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`runNondiffOptimization`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`runOptimization`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`runRequiredPasses`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`size`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`unpack`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`unpackReturnTuple`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)

### Includes

- **`ATen/core/ivalue.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`c10/util/Exception.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`c10/util/irange.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`cstdint`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`iterator`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`memory`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`mutex`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/autograd/edge.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/autograd/function.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/autograd/grad_mode.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/frontend/tracer.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/ir/ir.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/batch_mm.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/common_subexpression_elimination.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_pooling.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_propagation.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/create_autodiff_subgraphs.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/decompose_ops.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/graph_fuser.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/inline_autodiff_subgraphs.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/inliner.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/inplace_check.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/loop_unrolling.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/lower_grad_of.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/lower_tuples.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/pass_manager.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/peephole.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/remove_expands.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/remove_mutation.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/requires_grad_analysis.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/shape_analysis.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/specialize_autogradzero.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/passes/tensorexpr_fuser.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/python/update_graph_executor_opt.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/resource_guard.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/argument_spec.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/autodiff.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/custom_operator.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/graph_executor.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/graph_executor_impl.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/interpreter.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/logging.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/profiling_graph_executor_impl.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/profiling_record.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch/csrc/jit/runtime/simple_graph_executor_impl.h`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`unordered_map`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`utility`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`vector`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)

### Namespaces

- **`detail`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`static`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)
- **`torch`**: [graph_executor.cpp_docs.md](./graph_executor.cpp_docs.md)


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
