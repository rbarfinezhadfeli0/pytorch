# Documentation: `docs/torch/csrc/jit/passes/frozen_ops_to_mkldnn.cpp_kw.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/frozen_ops_to_mkldnn.cpp_kw.md`
- **Size**: 8,291 bytes (8.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/csrc/jit/passes/frozen_ops_to_mkldnn.cpp`

## File Information

- **Original File**: [torch/csrc/jit/passes/frozen_ops_to_mkldnn.cpp](../../../../../torch/csrc/jit/passes/frozen_ops_to_mkldnn.cpp)
- **Documentation**: [`frozen_ops_to_mkldnn.cpp_docs.md`](./frozen_ops_to_mkldnn.cpp_docs.md)
- **Folder**: `torch/csrc/jit/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`MKLDNNSubgraphSlicer`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`Subgraphs`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`all`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)

### Functions

- **`BroadOp`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ComputeSubgraphInMKLDNN`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ConstantMKLDNNTensorOp`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ConvertFrozenOpsToMKLDNN`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`InplaceMKLDNNSubgraph`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`MKLDNNGroupStart`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`MKLDNNLayerNormOp`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`aliasAnalysisFromSchema`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`assertNonTensorTypeDoesNotContainTensors`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`buildupSubgraphs`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`clamp_node_creator`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`computableInMKLDNN`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`computeSubgraphsInMKLDNN`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`containsMKLDNNGroup`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`frozenMkldnnCompatibleConvNode`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`frozenMkldnnCompatibleLinearNode`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`if`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`merge_sets`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`mkldnn_tensor_scalar_mul`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`moveConvWeightsToMKLDNN`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`moveWeightsToMKLDNN`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`nonConstantParameters`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`replaceInputWithMKLDNNTensor`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`run`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`shouldConsiderForMerge`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`supportedMKLDNNWeight`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`tensorInputIsMKLDNNSupported`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)

### Includes

- **`ATen/ATen.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/CPUFunctions.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/Config.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/Utils.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/core/stack.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/core/symbol.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/native/ConvUtils.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/native/layer_norm.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/native/mkldnn/MKLDNNCommon.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ATen/native/mkldnn/Utils.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`algorithm`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`c10/core/Layout.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`c10/core/ScalarType.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`c10/util/Exception.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`c10/util/StringUtil.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`c10/util/irange.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`dnnl_types.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`ideep.hpp`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`memory`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/ir/alias_analysis.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/ir/constants.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/ir/ir.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/common_subexpression_elimination.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_propagation.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/fold_conv_bn.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/frozen_conv_folding.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/frozen_ops_to_mkldnn.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/graph_rewrite_helper.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/peephole.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/remove_mutation.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/passes/utils/subgraph_utils.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/runtime/custom_operator.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/runtime/operator_options.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/types.h`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)

### Namespaces

- **`torch`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)
- **`void`**: [frozen_ops_to_mkldnn.cpp_docs.md](./frozen_ops_to_mkldnn.cpp_docs.md)


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

- **File Documentation**: `frozen_ops_to_mkldnn.cpp_kw.md_docs.md`
- **Keyword Index**: `frozen_ops_to_mkldnn.cpp_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
