# Keyword Index: `torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py`

## File Information

- **Original File**: [torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py](../../../../../torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py)
- **Documentation**: [`_fsdp_param_group.py_docs.md`](./_fsdp_param_group.py_docs.md)
- **Folder**: `torch/distributed/fsdp/_fully_shard`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AllGatherState`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`AllReduceState`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`FSDPCommContext`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`FSDPParamGroup`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`ReduceScatterState`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`RegisterPostBackwardFunction`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`represents`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)

### Functions

- **`__init__`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`__repr__`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_all_gather_process_group`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_all_reduce_process_group`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_assert_not_tracing_fsdp`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_backward_prefetch`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_get_param_module_infos`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_init_mp_dtypes`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_is_hsdp`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_prefetch_unshard`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_record_post_forward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_reduce_scatter_process_group`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_register_post_backward_hook`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_register_state_dict_hooks`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_reshard_after_forward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_to_sharded`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_to_sharded_post_forward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_to_unsharded`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_use_post_forward_mesh`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_validate_cpu_offload_params`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_validate_no_meta_params`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_wait_all_gather_streams_on_event`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_wait_for_post_backward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_with_fqn`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`backward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`finalize_backward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`forward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`get_all_gather_streams`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`is_sharded`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`is_sharded_post_forward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`is_unsharded`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`lazy_init`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`post_backward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`post_forward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`pre_backward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`pre_forward`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`reshard`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`set_allocate_memory_from_process_group`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`to_sharded_hook`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`unshard`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`use_training_state`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`wait_for_unshard`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)

### Imports

- **`._fsdp_api`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`._fsdp_collectives`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`._fsdp_common`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`._fsdp_param`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`Any`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`CPUOffloadPolicy`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`Callable`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`RemovableHandle`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`Shard`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_get_device_handle`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`_named_parameters_with_duplicates`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`alloc_storage`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`collections.abc`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`contextlib`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`logging`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`record_function`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch.distributed`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch.distributed.device_mesh`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch.distributed.fsdp._common_utils`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch.distributed.tensor`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch.nn`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch.profiler`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch.utils._pytree`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`torch.utils.hooks`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`tree_flatten`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)
- **`typing`**: [_fsdp_param_group.py_docs.md](./_fsdp_param_group.py_docs.md)


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
