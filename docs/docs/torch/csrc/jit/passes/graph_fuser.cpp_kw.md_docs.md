# Documentation: `docs/torch/csrc/jit/passes/graph_fuser.cpp_kw.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/graph_fuser.cpp_kw.md`
- **Size**: 5,218 bytes (5.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/csrc/jit/passes/graph_fuser.cpp`

## File Information

- **Original File**: [torch/csrc/jit/passes/graph_fuser.cpp](../../../../../torch/csrc/jit/passes/graph_fuser.cpp)
- **Documentation**: [`graph_fuser.cpp_docs.md`](./graph_fuser.cpp_docs.md)
- **Folder**: `torch/csrc/jit/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`GraphFuser`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)

### Functions

- **`FuseGraph`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`PeepholeOptimizeShapeExpressions`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`allUsersAreThisConsumerOrCalcSizes`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`calculatesSize`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`canFuseChunk`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`canFuseOnCPULegacy`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`canFuseWithConcat`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`for`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`fuseChunk`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`fuseChunkByReusingExistingFusedChunk`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`fuseConcats`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`insertExplicitBroadcast`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`isFusable`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`isFusableCatNode`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`isFusableDefault`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`isFusableDevice`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`isFusableMap`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`isSimpleMap`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`mergeFusionGroups`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`optimizeFusedGraphs`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`overrideCanFuseOnCPULegacy`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`removeOutputsUsedOnlyInSize`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`replaceIntermediateBroadcastingChunks`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`run`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`scanNodeForChunks`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`setInputArgLimit`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`sortReverseTopological`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`tensorInputs`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`tryToMoveChunk`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)

### Includes

- **`c10/util/Exception.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`c10/util/irange.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`queue`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/codegen/fuser/interface.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/frontend/ir_emitter.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/ir/alias_analysis.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/common_subexpression_elimination.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_pooling.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/graph_fuser.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/tensorexpr_fuser.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/passes/utils/subgraph_utils.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/runtime/autodiff.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/runtime/custom_operator.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch/csrc/jit/runtime/operator.h`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`unordered_map`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`utility`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)

### Namespaces

- **`static`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)
- **`torch`**: [graph_fuser.cpp_docs.md](./graph_fuser.cpp_docs.md)


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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`docs/torch/csrc/jit/passes`):

- [`peephole_dict_idioms.h_docs.md_docs.md`](./peephole_dict_idioms.h_docs.md_docs.md)
- [`remove_redundant_profiles.h_kw.md_docs.md`](./remove_redundant_profiles.h_kw.md_docs.md)
- [`loop_unrolling.cpp_kw.md_docs.md`](./loop_unrolling.cpp_kw.md_docs.md)
- [`onnx.h_kw.md_docs.md`](./onnx.h_kw.md_docs.md)
- [`guard_elimination.h_docs.md_docs.md`](./guard_elimination.h_docs.md_docs.md)
- [`frozen_conv_add_relu_fusion.cpp_docs.md_docs.md`](./frozen_conv_add_relu_fusion.cpp_docs.md_docs.md)
- [`hoist_conv_packed_params.h_kw.md_docs.md`](./hoist_conv_packed_params.h_kw.md_docs.md)
- [`lift_closures.h_kw.md_docs.md`](./lift_closures.h_kw.md_docs.md)
- [`frozen_conv_folding.h_kw.md_docs.md`](./frozen_conv_folding.h_kw.md_docs.md)
- [`frozen_graph_optimizations.h_docs.md_docs.md`](./frozen_graph_optimizations.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `graph_fuser.cpp_kw.md_docs.md`
- **Keyword Index**: `graph_fuser.cpp_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
