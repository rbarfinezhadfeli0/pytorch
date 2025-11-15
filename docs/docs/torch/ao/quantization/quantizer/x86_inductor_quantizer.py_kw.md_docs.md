# Documentation: `docs/torch/ao/quantization/quantizer/x86_inductor_quantizer.py_kw.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/quantizer/x86_inductor_quantizer.py_kw.md`
- **Size**: 10,082 bytes (9.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/ao/quantization/quantizer/x86_inductor_quantizer.py`

## File Information

- **Original File**: [torch/ao/quantization/quantizer/x86_inductor_quantizer.py](../../../../../torch/ao/quantization/quantizer/x86_inductor_quantizer.py)
- **Documentation**: [`x86_inductor_quantizer.py_docs.md`](./x86_inductor_quantizer.py_docs.md)
- **Folder**: `torch/ao/quantization/quantizer`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`X86InductorQuantizer`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`class`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`from`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)

### Functions

- **`__init__`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_cat`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_conv2d`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_conv2d_binary`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_conv2d_binary_unary`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_conv2d_fusion_pattern`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_conv2d_unary`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_conv_node_helper`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_linear`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_linear_binary_unary`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_linear_fusion_pattern`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_linear_node_helper`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_linear_unary`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_matmul`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_maxpool2d`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_nodes_not_quantize`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_output_for_int8_in_int8_out_pattern`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_output_for_int8_in_int8_out_pattern_entry`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_output_share_observer_as_input`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_propagation_quantizable_pattern`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_propagation_quantizable_pattern_entry`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_qat_conv2d_bn`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_qat_conv2d_bn_binary`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_qat_conv2d_bn_binary_unary`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_qat_conv2d_bn_unary`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_qat_conv2d_fusion_pattern`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_annotate_with_config`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_config_checker`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_create_module_name_filter`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_create_operator_type_filter`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_get_current_quantization_mode`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_get_input_idx_for_binary_node`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_get_output_nodes_of_partitions`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_global_config_filter`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_is_all_annotated`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_is_any_annotated`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_is_node_annotated`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_is_quantized_op_pt2e`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_map_module_function_to_aten_operator_type`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_mark_nodes_as_annotated`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_need_skip_config`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_set_aten_operator_qconfig`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_skip_annotate`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`annotate`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`check_all_nodes_from_module`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`get_default_x86_inductor_quantization_config`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`get_global_quantization_config`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`get_x86_inductor_linear_dynamic_fp16_config`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`is_all_inputs_connected_to_quantized_op`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`operator_type_filter`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`set_function_type_qconfig`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`set_global`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`set_module_name_qconfig`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`set_module_type_qconfig`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`validate`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`wrapper`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)

### Imports

- **`Any`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`Callable`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`Node`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_ObserverOrFakeQuantizeConstructor`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`_get_module_name_filter`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`collections.abc`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`dataclass`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`dataclasses`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`find_sequential_partitions`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`functools`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`itertools`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`operator`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.ao.quantization.fake_quantize`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.ao.quantization.observer`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.ao.quantization.pt2e.graph_utils`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.ao.quantization.quantizer.quantizer`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.ao.quantization.quantizer.utils`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.ao.quantization.quantizer.xnnpack_quantizer_utils`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.fx`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.fx.passes.utils.source_matcher_utils`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`torch.nn.functional`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`typing`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)
- **`warnings`**: [x86_inductor_quantizer.py_docs.md](./x86_inductor_quantizer.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/quantizer`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/quantizer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


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

Files in the same folder (`docs/torch/ao/quantization/quantizer`):

- [`xpu_inductor_quantizer.py_docs.md_docs.md`](./xpu_inductor_quantizer.py_docs.md_docs.md)
- [`xnnpack_quantizer_utils.py_kw.md_docs.md`](./xnnpack_quantizer_utils.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`embedding_quantizer.py_kw.md_docs.md`](./embedding_quantizer.py_kw.md_docs.md)
- [`embedding_quantizer.py_docs.md_docs.md`](./embedding_quantizer.py_docs.md_docs.md)
- [`composable_quantizer.py_docs.md_docs.md`](./composable_quantizer.py_docs.md_docs.md)
- [`xnnpack_quantizer_utils.py_docs.md_docs.md`](./xnnpack_quantizer_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`composable_quantizer.py_kw.md_docs.md`](./composable_quantizer.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `x86_inductor_quantizer.py_kw.md_docs.md`
- **Keyword Index**: `x86_inductor_quantizer.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
