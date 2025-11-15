# Keyword Index: `torch/ao/quantization/fx/utils.py`

## File Information

- **Original File**: [torch/ao/quantization/fx/utils.py](../../../../../torch/ao/quantization/fx/utils.py)
- **Documentation**: [`utils.py_docs.md`](./utils.py_docs.md)
- **Folder**: `torch/ao/quantization/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`class`**: [utils.py_docs.md](./utils.py_docs.md)
- **`from`**: [utils.py_docs.md](./utils.py_docs.md)

### Functions

- **`_activation_post_process_satisfies_dtype_config_constraints`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_module`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_observer_from_activation_post_process`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_insert_dequant_stub`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_insert_dequant_stubs_for_custom_module_lstm_output`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_is_custom_module_lstm`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_is_custom_module_mha`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_match_pattern`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_maybe_get_custom_module_lstm_from_node_arg`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_qconfig_satisfies_dtype_config_constraints`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_reroute_tuple_getitem_pattern`**: [utils.py_docs.md](./utils.py_docs.md)
- **`all_node_args_except_first`**: [utils.py_docs.md](./utils.py_docs.md)
- **`all_node_args_have_no_tensors`**: [utils.py_docs.md](./utils.py_docs.md)
- **`arg_indices_func`**: [utils.py_docs.md](./utils.py_docs.md)
- **`assert_and_get_unique_device`**: [utils.py_docs.md](./utils.py_docs.md)
- **`collect_producer_nodes`**: [utils.py_docs.md](./utils.py_docs.md)
- **`create_getattr_from_value`**: [utils.py_docs.md](./utils.py_docs.md)
- **`create_node_from_old_node_preserve_meta`**: [utils.py_docs.md](./utils.py_docs.md)
- **`find_patterns`**: [utils.py_docs.md](./utils.py_docs.md)
- **`forward`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_attr_name`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_custom_module_class_keys`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_linear_prepack_op_for_dtype`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_new_attr_name`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_new_attr_name_with_prefix`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_non_observable_arg_indexes_and_types`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_qconv_prepack_op`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_skipped_module_name_and_classes`**: [utils.py_docs.md](./utils.py_docs.md)
- **`graph_module_from_producer_nodes`**: [utils.py_docs.md](./utils.py_docs.md)
- **`load_arg`**: [utils.py_docs.md](./utils.py_docs.md)
- **`match_dq`**: [utils.py_docs.md](./utils.py_docs.md)
- **`match_getitem`**: [utils.py_docs.md](./utils.py_docs.md)
- **`match_lstm`**: [utils.py_docs.md](./utils.py_docs.md)
- **`match_tuple`**: [utils.py_docs.md](./utils.py_docs.md)
- **`maybe_get_next_module`**: [utils.py_docs.md](./utils.py_docs.md)
- **`node_arg_is_bias`**: [utils.py_docs.md](./utils.py_docs.md)
- **`node_arg_is_weight`**: [utils.py_docs.md](./utils.py_docs.md)
- **`return_arg_list`**: [utils.py_docs.md](./utils.py_docs.md)

### Imports

- **`._decomposed`**: [utils.py_docs.md](./utils.py_docs.md)
- **`.custom_config`**: [utils.py_docs.md](./utils.py_docs.md)
- **`Any`**: [utils.py_docs.md](./utils.py_docs.md)
- **`Callable`**: [utils.py_docs.md](./utils.py_docs.md)
- **`DTypeWithConstraints`**: [utils.py_docs.md](./utils.py_docs.md)
- **`DeQuantStub`**: [utils.py_docs.md](./utils.py_docs.md)
- **`Graph`**: [utils.py_docs.md](./utils.py_docs.md)
- **`GraphModule`**: [utils.py_docs.md](./utils.py_docs.md)
- **`PrepareCustomConfig`**: [utils.py_docs.md](./utils.py_docs.md)
- **`QConfigAny`**: [utils.py_docs.md](./utils.py_docs.md)
- **`QConfigMapping`**: [utils.py_docs.md](./utils.py_docs.md)
- **`collections`**: [utils.py_docs.md](./utils.py_docs.md)
- **`collections.abc`**: [utils.py_docs.md](./utils.py_docs.md)
- **`copy`**: [utils.py_docs.md](./utils.py_docs.md)
- **`dataclass`**: [utils.py_docs.md](./utils.py_docs.md)
- **`dataclasses`**: [utils.py_docs.md](./utils.py_docs.md)
- **`functools`**: [utils.py_docs.md](./utils.py_docs.md)
- **`namedtuple`**: [utils.py_docs.md](./utils.py_docs.md)
- **`operator`**: [utils.py_docs.md](./utils.py_docs.md)
- **`quantized_decomposed_lib`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.backend_config`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.fake_quantize`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.observer`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.qconfig_mapping`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.stubs`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.utils`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.fx`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.fx.graph`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.nn`**: [utils.py_docs.md](./utils.py_docs.md)
- **`typing`**: [utils.py_docs.md](./utils.py_docs.md)
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
