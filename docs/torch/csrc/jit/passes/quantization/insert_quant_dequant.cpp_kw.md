# Keyword Index: `torch/csrc/jit/passes/quantization/insert_quant_dequant.cpp`

## File Information

- **Original File**: [torch/csrc/jit/passes/quantization/insert_quant_dequant.cpp](../../../../../../torch/csrc/jit/passes/quantization/insert_quant_dequant.cpp)
- **Documentation**: [`insert_quant_dequant.cpp_docs.md`](./insert_quant_dequant.cpp_docs.md)
- **Folder**: `torch/csrc/jit/passes/quantization`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`InsertQuantDeQuantHelper`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`QuantOpParams`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`SubGraphCloneHelper`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)

### Functions

- **`InsertQuantDeQuant`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`InsertQuantDeQuantForOnDevicePTQ`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`RemoveRedundantDequantize`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`RemoveRedundantQuantizationOps`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`ReplicateChooseQParamsQuantDequant`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`ReplicateClampScalarArgs`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`ReplicateDeQuant`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`ReplicateQuant`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`back`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`checkCalculateQParamsResult`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`checkCalculateQParamsResultTypes`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`checkQScheme`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`for`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`getObserverDtype`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`if`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`insertChooseQParamQuantDequant`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`insertQuantizationOps`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`isEmbeddingBagOp`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`isPerChannel`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`isPlaceholderObserver`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`isQuantized`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`isWeight`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`removeDequantizeFromInputs`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`toAffine`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)

### Includes

- **`c10/core/QScheme.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`c10/util/irange.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`stack`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/frontend/schema_matching.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/ir/subgraph_matcher.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/passes/constant_propagation.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/passes/fuse_linear.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/passes/graph_rewrite_helper.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/passes/inliner.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/passes/quantization/helper.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/passes/quantization/insert_quant_dequant.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`torch/csrc/jit/passes/subgraph_rewrite.h`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`utility`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)

### Namespaces

- **`torch`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)
- **`void`**: [insert_quant_dequant.cpp_docs.md](./insert_quant_dequant.cpp_docs.md)


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
