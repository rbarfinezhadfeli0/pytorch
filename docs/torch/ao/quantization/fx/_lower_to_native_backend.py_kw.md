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
