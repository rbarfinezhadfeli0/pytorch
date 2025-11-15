# Keyword Index: `torch/ao/quantization/quantizer/xnnpack_quantizer_utils.py`

## File Information

- **Original File**: [torch/ao/quantization/quantizer/xnnpack_quantizer_utils.py](../../../../../torch/ao/quantization/quantizer/xnnpack_quantizer_utils.py)
- **Documentation**: [`xnnpack_quantizer_utils.py_docs.md`](./xnnpack_quantizer_utils.py_docs.md)
- **Folder**: `torch/ao/quantization/quantizer`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`OperatorConfig`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`QuantizationConfig`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`from`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)

### Functions

- **`_annotate_adaptive_avg_pool2d`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_add`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_add_relu`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_cat`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_conv`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_conv_bn`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_conv_bn_relu`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_conv_relu`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_conv_transpose_bn`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_conv_transpose_bn_relu`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_conv_transpose_relu`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_gru_io_only`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_linear`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_linear_relu`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_mul`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_annotate_mul_relu`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_conv_bn`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_convert_scalars_to_attrs`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_do_annotate_conv_bn`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_do_annotate_conv_relu`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_is_annotated`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_is_input_large_scalar`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_is_input_non_float_tensor`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_is_share_obs_or_fq_op`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_mark_nodes_as_annotated`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`decorator`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`get_bias_qspec`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`get_input_act_qspec`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`get_output_act_qspec`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`get_pattern`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`get_weight_qspec`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`propagate_annotation`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`register_annotator`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)

### Imports

- **`Callable`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`FakeTensor`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`NamedTuple`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`Node`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`_WrapperModule`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`collections.abc`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`dataclass`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`dataclasses`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`get_new_attr_name_with_prefix`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`get_source_partitions`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`itertools`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch._subclasses`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.ao.quantization.fx.utils`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.ao.quantization.pt2e.export_utils`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.ao.quantization.pt2e.utils`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.ao.quantization.quantizer`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.ao.quantization.quantizer.utils`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.fx`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.fx.passes.utils.matcher_with_name_node_map_utils`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.fx.passes.utils.source_matcher_utils`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`torch.nn.functional`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)
- **`typing`**: [xnnpack_quantizer_utils.py_docs.md](./xnnpack_quantizer_utils.py_docs.md)


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
