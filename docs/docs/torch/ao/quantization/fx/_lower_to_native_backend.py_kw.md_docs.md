# Documentation: `docs/torch/ao/quantization/fx/_lower_to_native_backend.py_kw.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/fx/_lower_to_native_backend.py_kw.md`
- **Size**: 7,539 bytes (7.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/ao/quantization/fx/_lower_to_native_backend.py`

## File Information

- **Original File**: [torch/ao/quantization/fx/_lower_to_native_backend.py](../../../../../torch/ao/quantization/fx/_lower_to_native_backend.py)
- **Documentation**: [`_lower_to_native_backend.py_docs.md`](./_lower_to_native_backend.py_docs.md)
- **Folder**: `torch/ao/quantization/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`for`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`in`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`of`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`to`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)

### Functions

- **`_get_module`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_is_node_in_list`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_load_packed_weight`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_dynamic_weighted_ref_functional`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_dynamic_weighted_ref_module`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_get_tensor_info_op`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_getattr_tensor_metadta_op`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_quantized_binary_op`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_static_weighted_ref_functional`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_static_weighted_ref_module`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_static_weighted_ref_module_with_two_inputs`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_to_native_backend`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_lower_weight_only_weighted_ref_module`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_match_static_pattern`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_match_static_pattern_with_two_inputs`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_save_packed_weight`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`fold_weight`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_copy_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_default_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_dequantize_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_fixed_qparams_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_general_tensor_shape_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_get_tensor_info_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_getattr_tensor_metadata_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_other_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`is_special_pattern_node`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`load_arg`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`should_skip_lowering`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`special_pattern_replacement`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)

### Imports

- **`.utils`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`Any`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`Callable`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`Graph`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`GraphModule`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`QConfigAny`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`WeightedQuantizedModule`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`_parent_name`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`collections.abc`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`get_quantized_operator`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`operator`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.nn.intrinsic`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.nn.intrinsic.quantized`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.nn.intrinsic.quantized.dynamic`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.nn.quantized`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.nn.quantized.dynamic`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.nn.quantized.modules.utils`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.nn.quantized.reference`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.quantization.quantization_mappings`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.ao.quantization.utils`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.fx`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.fx.graph`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.nn`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`torch.nn.functional`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)
- **`typing`**: [_lower_to_native_backend.py_docs.md](./_lower_to_native_backend.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/fx`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/ao/quantization/fx`):

- [`fuse_handler.py_docs.md_docs.md`](./fuse_handler.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`quantize_handler.py_kw.md_docs.md`](./quantize_handler.py_kw.md_docs.md)
- [`lstm_utils.py_kw.md_docs.md`](./lstm_utils.py_kw.md_docs.md)
- [`prepare.py_kw.md_docs.md`](./prepare.py_kw.md_docs.md)
- [`graph_module.py_docs.md_docs.md`](./graph_module.py_docs.md_docs.md)
- [`fuse_handler.py_kw.md_docs.md`](./fuse_handler.py_kw.md_docs.md)
- [`quantize_handler.py_docs.md_docs.md`](./quantize_handler.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`lower_to_qnnpack.py_kw.md_docs.md`](./lower_to_qnnpack.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_lower_to_native_backend.py_kw.md_docs.md`
- **Keyword Index**: `_lower_to_native_backend.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
