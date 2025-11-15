# Documentation: `docs/torch/_export/passes/replace_quantized_ops_with_standard_ops_pass.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_export/passes/replace_quantized_ops_with_standard_ops_pass.py_kw.md`
- **Size**: 7,194 bytes (7.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_export/passes/replace_quantized_ops_with_standard_ops_pass.py`

## File Information

- **Original File**: [torch/_export/passes/replace_quantized_ops_with_standard_ops_pass.py](../../../../torch/_export/passes/replace_quantized_ops_with_standard_ops_pass.py)
- **Documentation**: [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **Folder**: `torch/_export/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GraphModule`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)

### Functions

- **`_clean_attr`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_conv1d_op_with_squeeze`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_transform_batch_norm`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_transform_conv_with_packedparam`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_transform_linear_with_packedparam`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_transform_op_where_last_two_arguments_are_scale_and_zero_point`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_transform_prepacked_op`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_transform_scalar_arithmetic`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`forward`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`fx_enum_to_dtype`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`fx_transform_quantized_op_to_standard_op`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`get_dequantized`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`get_qmin_qmax`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`get_script_object`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`get_tensor_from_qtensor`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`insert_dequantized_node`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`insert_fused_activation_node`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`insert_qmin_qmax_node`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`insert_quantized_node`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`insert_weight_and_bias_get_attr_node`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`insert_weight_and_bias_get_attr_node_from_get_attr_to_qtensor`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`insert_weight_and_bias_get_attr_node_from_get_attr_to_scriptobject`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`int_to_valid_dtype`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`replace_quantized_ops_with_standard_ops`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)

### Imports

- **`OpOverload`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`Optional`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_TORCH_ENUM_TO_DTYPE`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`_assign_attr`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`calculate_qmin_qmax`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`logging`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`operator`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`torch`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`torch._export.converter`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`torch._ops`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`torch.ao.quantization.fx._decomposed`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`torch.ao.quantization.utils`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`torch.export._trace`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`torch.fx.graph_module`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- **`typing`**: [replace_quantized_ops_with_standard_ops_pass.py_docs.md](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_export/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/_export/passes`):

- [`replace_set_grad_with_hop_pass.py_docs.md_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md_docs.md)
- [`_node_metadata_hook.py_docs.md_docs.md`](./_node_metadata_hook.py_docs.md_docs.md)
- [`replace_view_ops_with_view_copy_ops_pass.py_kw.md_docs.md`](./replace_view_ops_with_view_copy_ops_pass.py_kw.md_docs.md)
- [`lift_constants_pass.py_kw.md_docs.md`](./lift_constants_pass.py_kw.md_docs.md)
- [`remove_runtime_assertions.py_kw.md_docs.md`](./remove_runtime_assertions.py_kw.md_docs.md)
- [`lift_constants_pass.py_docs.md_docs.md`](./lift_constants_pass.py_docs.md_docs.md)
- [`constant_folding.py_docs.md_docs.md`](./constant_folding.py_docs.md_docs.md)
- [`remove_runtime_assertions.py_docs.md_docs.md`](./remove_runtime_assertions.py_docs.md_docs.md)
- [`constant_folding.py_kw.md_docs.md`](./constant_folding.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md`
- **Keyword Index**: `replace_quantized_ops_with_standard_ops_pass.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
