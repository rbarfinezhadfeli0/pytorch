# Keyword Index: `torch/csrc/jit/runtime/profiling_graph_executor_impl.cpp`

## File Information

- **Original File**: [torch/csrc/jit/runtime/profiling_graph_executor_impl.cpp](../../../../../torch/csrc/jit/runtime/profiling_graph_executor_impl.cpp)
- **Documentation**: [`profiling_graph_executor_impl.cpp_docs.md`](./profiling_graph_executor_impl.cpp_docs.md)
- **Folder**: `torch/csrc/jit/runtime`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`getBailoutDepth`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`getFusionStrategy`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`getInitialStrategy`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`getNowInSecs`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`guardDifferentiableGraph`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`if`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`needsGradientInProfilingMode`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`runNooptPassPipeline`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`runPreAutodiffPassPipeline`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`setFusionStrategy`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`setRequiresGradOnDiffGraph`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)

### Includes

- **`c10/util/irange.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`chrono`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`mutex`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`optional`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/add_if_then_else.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/bailout_graph.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/batch_mm.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/check_strict_fusion.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/clear_profiling.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/clear_undefinedness.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/common_subexpression_elimination.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_pooling.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_propagation.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/create_autodiff_subgraphs.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/decompose_ops.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/graph_fuser.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/guard_elimination.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/inline_autodiff_subgraphs.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/inliner.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/inplace_check.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/insert_guards.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/loop_unrolling.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/lower_grad_of.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/lower_tuples.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/pass_manager.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/peephole.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/remove_expands.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/remove_mutation.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/requires_grad_analysis.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/shape_analysis.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/specialize_autogradzero.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/tensorexpr_fuser.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/passes/utils/subgraph_utils.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)
- **`torch/csrc/jit/runtime/profiling_graph_executor_impl.h`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)

### Namespaces

- **`torch`**: [profiling_graph_executor_impl.cpp_docs.md](./profiling_graph_executor_impl.cpp_docs.md)


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
