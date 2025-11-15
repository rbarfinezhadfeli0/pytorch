# Keyword Index: `torch/csrc/jit/runtime/static/impl.cpp`

## File Information

- **Original File**: [torch/csrc/jit/runtime/static/impl.cpp](../../../../../../torch/csrc/jit/runtime/static/impl.cpp)
- **Documentation**: [`impl.cpp_docs.md`](./impl.cpp_docs.md)
- **Folder**: `torch/csrc/jit/runtime/static`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`planner_`**: [impl.cpp_docs.md](./impl.cpp_docs.md)

### Functions

- **`IsSelfInGraphInput`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`OptimizeGraph`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`PrepareGraphForStaticModule`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`allArgsAreTensors`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`canEnableStaticRuntime`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`canEnableStaticRuntimeImpl`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`checkNoMemoryOverlap`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`check_type`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`containTensorsOnly`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`destroyNodeOutputs`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`display_ivalue`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`display_pnode_info`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`dumpValueSet`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`escapesScope`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`for`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`generate_latency_json`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`iValueToString`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`if`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`isPureFunction`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`isTensorList`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`isUnsupportedOp`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`mayContainAlias`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`removeSelfFromGraphInput`**: [impl.cpp_docs.md](./impl.cpp_docs.md)

### Includes

- **`ATen/MemoryOverlap.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`ATen/core/symbol.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`ATen/ops/clone_native.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`ATen/record_function.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`algorithm`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`c10/core/CPUAllocator.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`c10/core/InferenceMode.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`c10/macros/Macros.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`c10/util/MaybeOwned.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`c10/util/irange.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`caffe2/core/timer.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`common/logging/logging.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`cstdint`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`folly/dynamic.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`folly/json.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`iostream`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`iterator`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`limits`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`sstream`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/ir/alias_analysis.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/passes/add_if_then_else.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/passes/canonicalize.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/passes/eliminate_no_ops.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/passes/freeze_module.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/passes/remove_mutation.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/passes/subgraph_rewrite.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/passes/variadic_ops.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/runtime/graph_iterator.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/runtime/static/fusion.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/runtime/static/impl.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/runtime/static/memory_planner.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/runtime/static/ops.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/runtime/static/passes.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch/csrc/jit/runtime/vararg_functions.h`**: [impl.cpp_docs.md](./impl.cpp_docs.md)

### Namespaces

- **`std`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`template`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`torch`**: [impl.cpp_docs.md](./impl.cpp_docs.md)
- **`void`**: [impl.cpp_docs.md](./impl.cpp_docs.md)


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
