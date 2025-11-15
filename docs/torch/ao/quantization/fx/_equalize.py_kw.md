# Keyword Index: `torch/ao/quantization/fx/_equalize.py`

## File Information

- **Original File**: [torch/ao/quantization/fx/_equalize.py](../../../../../torch/ao/quantization/fx/_equalize.py)
- **Documentation**: [`_equalize.py_docs.md`](./_equalize.py_docs.md)
- **Folder**: `torch/ao/quantization/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`EqualizationQConfig`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`_InputEqualizationObserver`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`_WeightEqualizationObserver`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`instead`**: [_equalize.py_docs.md](./_equalize.py_docs.md)

### Functions

- **`__init__`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`__new__`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`_convert_equalization_ref`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`calculate_equalization_scale`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`calculate_scaled_minmax`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`clear_weight_quant_obs_node`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`convert_eq_obs`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`custom_module_supports_equalization`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`forward`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`fused_module_supports_equalization`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`get_equalization_qconfig_dict`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`get_input_minmax`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`get_layer_sqnr_dict`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`get_op_node_and_weight_eq_obs`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`get_weight_col_minmax`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`is_equalization_observer`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`maybe_get_next_equalization_scale`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`maybe_get_next_input_eq_obs`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`maybe_get_weight_eq_obs_node`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`nn_module_supports_equalization`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`node_supports_equalization`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`remove_node`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`reshape_scale`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`scale_input_observer`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`scale_weight_functional`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`scale_weight_node`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`set_equalization_scale`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`update_obs_for_equalization`**: [_equalize.py_docs.md](./_equalize.py_docs.md)

### Imports

- **`.utils`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`Any`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`GraphModule`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`Node`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`_get_observed_graph_module_attr`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`_parent_name`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`collections`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`get_unmatchable_types_map`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`namedtuple`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`operator`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.ao.nn.intrinsic`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.ao.ns._numeric_suite_fx`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.ao.ns.fx.mappings`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.ao.quantization.fx.graph_module`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.ao.quantization.observer`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.ao.quantization.utils`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.fx`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.fx.graph`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.nn`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`torch.nn.functional`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`typing`**: [_equalize.py_docs.md](./_equalize.py_docs.md)
- **`warnings`**: [_equalize.py_docs.md](./_equalize.py_docs.md)


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
