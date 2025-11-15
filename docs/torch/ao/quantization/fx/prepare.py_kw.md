# Keyword Index: `torch/ao/quantization/fx/prepare.py`

## File Information

- **Original File**: [torch/ao/quantization/fx/prepare.py](../../../../../torch/ao/quantization/fx/prepare.py)
- **Documentation**: [`prepare.py_docs.md`](./prepare.py_docs.md)
- **Folder**: `torch/ao/quantization/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`soon`**: [prepare.py_docs.md](./prepare.py_docs.md)

### Functions

- **`_add_matched_node_name_to_set`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_create_obs_or_fq_from_qspec`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_get_arg_as_input_act_obs_or_fq`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_get_arg_target_dtype_as_output`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_get_dtype_and_is_dynamic`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_get_observer_kwargs`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_get_output_act_obs_or_fq`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_get_qspec_for_arg`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_get_standalone_module_configs`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_get_target_activation_dtype_for_node`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_insert_obs_or_fq`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_is_activation_post_process_node`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_is_input_arg_dtype_supported_by_backend`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_is_observer_in_same_graph`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_is_output_dtype_supported_by_backend`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_is_pattern_dtype_config_and_qconfig_supported_by_backend`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_maybe_insert_input_equalization_observers_for_node`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_maybe_insert_input_observer_for_arg_or_kwarg`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_maybe_insert_input_observers_for_node`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_maybe_insert_observers_before_graph_output`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_maybe_insert_output_observer_for_node`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_maybe_make_input_output_share_observers`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_maybe_propagate_dtype_for_node`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_needs_obs_or_fq`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_qat_swap_modules`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_recursive_maybe_replace_node_with_obs`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_remove_output_observer`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_run_prepare_fx_on_standalone_modules`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_save_state`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_set_target_dtype_info_for_matched_node_pattern`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_swap_custom_module_to_observed`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`insert_observers_for_model`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`prepare`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`propagate_dtypes_for_known_nodes`**: [prepare.py_docs.md](./prepare.py_docs.md)

### Imports

- **`._equalize`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`.custom_config`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`.match_utils`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`.pattern_utils`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`.qconfig_mapping_utils`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`.quantize_handler`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`.utils`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`Any`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`Argument`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`FakeTensor`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`Graph`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`GraphModule`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`PrepareCustomConfig`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`QConfigMapping`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_find_matches`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_is_activation_post_process`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_is_reuse_input_qconfig`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`_sorted_patterns_dict`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`asdict`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`convert`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`copy`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`dataclasses`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`is_equalization_observer`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch._subclasses`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization.backend_config`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization.backend_config.utils`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization.observer`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization.qconfig_mapping`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization.quantize`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization.quantizer`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.ao.quantization.utils`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.fx`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.fx.graph`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`torch.fx.node`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`typing`**: [prepare.py_docs.md](./prepare.py_docs.md)
- **`warnings`**: [prepare.py_docs.md](./prepare.py_docs.md)


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
