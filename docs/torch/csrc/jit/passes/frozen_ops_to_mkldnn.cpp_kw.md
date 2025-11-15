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
