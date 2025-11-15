# Keyword Index: `torch/onnx/_internal/torchscript_exporter/utils.py`

## File Information

- **Original File**: [torch/onnx/_internal/torchscript_exporter/utils.py](../../../../../torch/onnx/_internal/torchscript_exporter/utils.py)
- **Documentation**: [`utils.py_docs.md`](./utils.py_docs.md)
- **Folder**: `torch/onnx/_internal/torchscript_exporter`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`SumModule`**: [utils.py_docs.md](./utils.py_docs.md)
- **`but`**: [utils.py_docs.md](./utils.py_docs.md)
- **`variables`**: [utils.py_docs.md](./utils.py_docs.md)

### Functions

- **`__register_attribute_hook`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_add_block`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_add_input_to_block`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_add_output_to_block`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_apply_friendly_debug_names`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_check_flatten_did_not_remove`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_create_jit_graph`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_decide_add_node_names`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_decide_constant_folding`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_decide_input_format`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_decide_keep_init_as_input`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_export`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_find_typename`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_aten_op_overload_name`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_example_outputs`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_module_attributes`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_named_param_dict`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_param_count_list`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_is_constant_tensor_list`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_model_to_graph`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_optimize_graph`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_pre_trace_quant_model`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_reset_trace_module_map`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_resolve_args_by_export_type`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_run_symbolic_function`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_run_symbolic_method`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_set_input_and_output_names`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_setup_trace_module_map`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_should_aten_fallback`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_signature`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_split_tensor_list_constants`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_trace`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_trace_and_get_graph_from_model`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_track_module_attributes_forward_hook`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_track_module_attributes_forward_pre_hook`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_trigger_symbolic_function_registration`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_unqualified_variable_name`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_validate_dynamic_axes`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_verify_custom_op_name`**: [utils.py_docs.md](./utils.py_docs.md)
- **`disable_apex_o2_state_dict_hook`**: [utils.py_docs.md](./utils.py_docs.md)
- **`export`**: [utils.py_docs.md](./utils.py_docs.md)
- **`exporter_context`**: [utils.py_docs.md](./utils.py_docs.md)
- **`flatten`**: [utils.py_docs.md](./utils.py_docs.md)
- **`forward`**: [utils.py_docs.md](./utils.py_docs.md)
- **`model_signature`**: [utils.py_docs.md](./utils.py_docs.md)
- **`register_custom_op_symbolic`**: [utils.py_docs.md](./utils.py_docs.md)
- **`select_model_mode_for_export`**: [utils.py_docs.md](./utils.py_docs.md)
- **`set_names`**: [utils.py_docs.md](./utils.py_docs.md)
- **`setup_onnx_logging`**: [utils.py_docs.md](./utils.py_docs.md)
- **`unconvertible_ops`**: [utils.py_docs.md](./utils.py_docs.md)
- **`unpack_quantized_tensor`**: [utils.py_docs.md](./utils.py_docs.md)
- **`unregister_custom_op_symbolic`**: [utils.py_docs.md](./utils.py_docs.md)
- **`warn_on_static_input_change`**: [utils.py_docs.md](./utils.py_docs.md)

### Imports

- **`Any`**: [utils.py_docs.md](./utils.py_docs.md)
- **`Callable`**: [utils.py_docs.md](./utils.py_docs.md)
- **`GLOBALS`**: [utils.py_docs.md](./utils.py_docs.md)
- **`IValue`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_C`**: [utils.py_docs.md](./utils.py_docs.md)
- **`__future__`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_constants`**: [utils.py_docs.md](./utils.py_docs.md)
- **`annotations`**: [utils.py_docs.md](./utils.py_docs.md)
- **`collections.abc`**: [utils.py_docs.md](./utils.py_docs.md)
- **`contextlib`**: [utils.py_docs.md](./utils.py_docs.md)
- **`copy`**: [utils.py_docs.md](./utils.py_docs.md)
- **`deprecated`**: [utils.py_docs.md](./utils.py_docs.md)
- **`inspect`**: [utils.py_docs.md](./utils.py_docs.md)
- **`re`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch._C._onnx`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.jit._trace`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.onnx`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.onnx._internal.torchscript_exporter`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.onnx._internal.torchscript_exporter._globals`**: [utils.py_docs.md](./utils.py_docs.md)
- **`typing`**: [utils.py_docs.md](./utils.py_docs.md)
- **`typing_extensions`**: [utils.py_docs.md](./utils.py_docs.md)
- **`warnings`**: [utils.py_docs.md](./utils.py_docs.md)


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
