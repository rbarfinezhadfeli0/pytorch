# Keyword Index: `torch/distributed/fsdp/_common_utils.py`

## File Information

- **Original File**: [torch/distributed/fsdp/_common_utils.py](../../../../torch/distributed/fsdp/_common_utils.py)
- **Documentation**: [`_common_utils.py_docs.md`](./_common_utils.py_docs.md)
- **Folder**: `torch/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`HandleTrainingState`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`TrainingState`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_FSDPDeviceHandle`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_FSDPState`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_UninitializedDeviceHandle`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`to`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)

### Functions

- **`__getattr__`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`__getattribute__`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`__init__`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_apply_to_modules`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_assert_in_training_states`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_get_handle_fqns_from_root`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_get_module_fsdp_state`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_get_module_fsdp_state_if_fully_sharded_module`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_get_param_to_fqns`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_get_root_modules`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_get_sharding_strategy`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_has_fsdp_params`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_is_composable`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_is_fsdp_flattened`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_log_post_backward_hook`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_module_handle`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_named_parameters_with_duplicates`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_no_dispatch_record_stream`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_override_module_mixed_precision`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_set_fsdp_flattened`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`cast_fn`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`clean_tensor_name`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`f`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`forward_post_hook`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`forward_pre_hook`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`from_device`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`module_fn`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`return_fn`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)

### Imports

- **`._flat_param`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`.api`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`Any`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`Callable`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`DeviceMesh`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`FSDPExtensions`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`FlatParamHandle`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_apply_to_tensors`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`_get_module_state`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`auto`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`chain`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`collections.abc`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`enum`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`functools`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`itertools`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`logging`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`no_dispatch`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`partial`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.distributed`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.distributed._composable_state`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.distributed.device_mesh`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.distributed.fsdp._flat_param`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.distributed.fsdp._fsdp_extensions`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.distributed.utils`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.nn`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`torch.utils._mode_utils`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`traceback`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`typing`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`warnings`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)
- **`weakref`**: [_common_utils.py_docs.md](./_common_utils.py_docs.md)


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
