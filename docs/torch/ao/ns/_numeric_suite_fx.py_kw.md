# Keyword Index: `torch/ao/ns/_numeric_suite_fx.py`

## File Information

- **Original File**: [torch/ao/ns/_numeric_suite_fx.py](../../../../torch/ao/ns/_numeric_suite_fx.py)
- **Documentation**: [`_numeric_suite_fx.py_docs.md`](./_numeric_suite_fx.py_docs.md)
- **Folder**: `torch/ao/ns`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`NSTracer`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`OutputComparisonLogger`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`OutputLogger`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`for`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`if`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`of`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)

### Functions

- **`__init__`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`__repr__`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_add_loggers_impl`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_add_loggers_one_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_add_shadow_loggers_impl`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_extract_logger_info_one_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_extract_weights_impl`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_extract_weights_one_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_n_shadows_compare_weights`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_prepare_n_shadows_add_loggers_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`add_loggers`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`add_shadow_loggers`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`convert_n_shadows_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`extend_logger_results_with_comparison`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`extract_logger_info`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`extract_results_n_shadows_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`extract_shadow_logger_info`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`extract_weights`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`forward`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`is_leaf_module`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`loggers_set_enabled`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`loggers_set_save_activations`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`prepare_n_shadows_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`print_comparisons_n_shadows_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)

### Imports

- **`.fx.graph_passes`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`.fx.ns_types`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`.fx.utils`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`.fx.weight_utils`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`Any`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`BackendConfig`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`Callable`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`GraphModule`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`NSNodeTargetType`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`Node`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`QConfigAny`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`QConfigMapping`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`QConfigMultiMapping`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_find_matches`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_get_observed_graph_module_attr`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`_get_pattern_to_quantize_handlers`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`add_loggers_to_model`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`collections`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`collections.abc`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`copy`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`extract_weight_from_node`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`get_base_name_to_sets_of_related_ops`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`get_matching_subgraph_pairs`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.ns._numeric_suite_fx`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.ns.fx.graph_matcher`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.ns.fx.mappings`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.ns.fx.n_shadows_utils`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.ns.fx.qconfig_multi_mapping`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization.backend_config`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization.backend_config.utils`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization.fx.graph_module`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization.fx.match_utils`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization.fx.qconfig_mapping_utils`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization.fx.quantize_handler`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.ao.quantization.quantize_fx`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.fx`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.fx.graph`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`torch.nn`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)
- **`typing`**: [_numeric_suite_fx.py_docs.md](./_numeric_suite_fx.py_docs.md)


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
