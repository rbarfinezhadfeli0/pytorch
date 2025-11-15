# Keyword Index: `torch/ao/quantization/pt2e/qat_utils.py`

## File Information

- **Original File**: [torch/ao/quantization/pt2e/qat_utils.py](../../../../../torch/ao/quantization/pt2e/qat_utils.py)
- **Documentation**: [`qat_utils.py_docs.md`](./qat_utils.py_docs.md)
- **Folder**: `torch/ao/quantization/pt2e`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_append_qdq`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_conv_bn_pattern`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_copy_over_literal_conv_args`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_copy_over_q_dq_args`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_duplicate_dequantize_node`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_filter_nodes_map`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_fold_conv_bn_qat`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_fold_conv_bn_qat_helper`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_folded_quantized_qat_conv_bn_pattern`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_fuse_conv_bn_qat`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_fuse_conv_bn_qat_helper`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_conv_bn_pattern`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_conv_bn_pattern_nodes`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_folded_quantized_qat_conv_bn_pattern`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_new_edge_or_node`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_new_qspec`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_nodes`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_q_dq_nodes`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_qat_conv_bn_pattern`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_qat_conv_bn_pattern_no_conv_bias`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_quantized_conv_bn_example_inputs_kwargs`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_get_quantized_qat_conv_bn_pattern`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_has_conv_bias_filter`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_is_dequantize`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_is_quantize`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_no_conv_bias_filter`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_qat_conv_bn_pattern`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_qat_conv_bn_pattern_no_conv_bias`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_quantized_qat_conv_bn_pattern`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_remove_extra_dequantize`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_update_conv_input_qspec_map_after_replacement`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_update_special_qspecs_after_replacement`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)

### Imports

- **`.utils`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`Any`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`Callable`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`Graph`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`InternalMatch`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`_WrapperModule`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`collections.abc`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`copy`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`dataclasses`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`itertools`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`operator`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`quantized_decomposed_lib`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`replace_pattern_with_filters`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`torch`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`torch.ao.quantization.fx._decomposed`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`torch.ao.quantization.pt2e.export_utils`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`torch.ao.quantization.quantizer`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`torch.fx`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`torch.fx.passes.utils.matcher_with_name_node_map_utils`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`torch.fx.subgraph_rewriter`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`torch.nn.functional`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)
- **`typing`**: [qat_utils.py_docs.md](./qat_utils.py_docs.md)


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
