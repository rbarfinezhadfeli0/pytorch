# Keyword Index: `torch/ao/quantization/utils.py`

## File Information

- **Original File**: [torch/ao/quantization/utils.py](../../../../torch/ao/quantization/utils.py)
- **Documentation**: [`utils.py_docs.md`](./utils.py_docs.md)
- **Folder**: `torch/ao/quantization`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`M`**: [utils.py_docs.md](./utils.py_docs.md)
- **`MatchAllNode`**: [utils.py_docs.md](./utils.py_docs.md)
- **`for`**: [utils.py_docs.md](./utils.py_docs.md)
- **`instance`**: [utils.py_docs.md](./utils.py_docs.md)
- **`mapping`**: [utils.py_docs.md](./utils.py_docs.md)
- **`that`**: [utils.py_docs.md](./utils.py_docs.md)

### Functions

- **`__init__`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_assert_and_get_unique_device`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_default_kwargs`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_num_pos_args`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_path_of_module`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_signature_locals`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_normalize_kwargs`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_parent_name`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_patched_module_call`**: [utils.py_docs.md](./utils.py_docs.md)
- **`activation_dtype`**: [utils.py_docs.md](./utils.py_docs.md)
- **`activation_is_dynamically_quantized`**: [utils.py_docs.md](./utils.py_docs.md)
- **`activation_is_int32_quantized`**: [utils.py_docs.md](./utils.py_docs.md)
- **`activation_is_int8_quantized`**: [utils.py_docs.md](./utils.py_docs.md)
- **`activation_is_statically_quantized`**: [utils.py_docs.md](./utils.py_docs.md)
- **`calculate_qmin_qmax`**: [utils.py_docs.md](./utils.py_docs.md)
- **`check_min_max_valid`**: [utils.py_docs.md](./utils.py_docs.md)
- **`check_node`**: [utils.py_docs.md](./utils.py_docs.md)
- **`determine_qparams`**: [utils.py_docs.md](./utils.py_docs.md)
- **`f`**: [utils.py_docs.md](./utils.py_docs.md)
- **`forward`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_combined_dict`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_fqn_to_example_inputs`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_qconfig_dtypes`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_qparam_dict`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_quant_type`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_swapped_custom_module_class`**: [utils.py_docs.md](./utils.py_docs.md)
- **`getattr_from_fqn`**: [utils.py_docs.md](./utils.py_docs.md)
- **`has_no_children_ignoring_parametrizations`**: [utils.py_docs.md](./utils.py_docs.md)
- **`is_per_channel`**: [utils.py_docs.md](./utils.py_docs.md)
- **`is_per_tensor`**: [utils.py_docs.md](./utils.py_docs.md)
- **`op_is_int8_dynamically_quantized`**: [utils.py_docs.md](./utils.py_docs.md)
- **`to_underlying_dtype`**: [utils.py_docs.md](./utils.py_docs.md)
- **`validate_qmin_qmax`**: [utils.py_docs.md](./utils.py_docs.md)
- **`weight_dtype`**: [utils.py_docs.md](./utils.py_docs.md)
- **`weight_is_quantized`**: [utils.py_docs.md](./utils.py_docs.md)
- **`weight_is_statically_quantized`**: [utils.py_docs.md](./utils.py_docs.md)

### Imports

- **`Any`**: [utils.py_docs.md](./utils.py_docs.md)
- **`Callable`**: [utils.py_docs.md](./utils.py_docs.md)
- **`Node`**: [utils.py_docs.md](./utils.py_docs.md)
- **`OrderedDict`**: [utils.py_docs.md](./utils.py_docs.md)
- **`PlaceholderObserver`**: [utils.py_docs.md](./utils.py_docs.md)
- **`QuantType`**: [utils.py_docs.md](./utils.py_docs.md)
- **`TypeAliasType`**: [utils.py_docs.md](./utils.py_docs.md)
- **`collections`**: [utils.py_docs.md](./utils.py_docs.md)
- **`collections.abc`**: [utils.py_docs.md](./utils.py_docs.md)
- **`functools`**: [utils.py_docs.md](./utils.py_docs.md)
- **`getfullargspec`**: [utils.py_docs.md](./utils.py_docs.md)
- **`inspect`**: [utils.py_docs.md](./utils.py_docs.md)
- **`is_parametrized`**: [utils.py_docs.md](./utils.py_docs.md)
- **`sys`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.observer`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.ao.quantization.quant_type`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.fx`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.nn.utils.parametrize`**: [utils.py_docs.md](./utils.py_docs.md)
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
