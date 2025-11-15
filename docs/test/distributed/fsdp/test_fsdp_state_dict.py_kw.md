# Keyword Index: `test/distributed/fsdp/test_fsdp_state_dict.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_state_dict.py](../../../../test/distributed/fsdp/test_fsdp_state_dict.py)
- **Documentation**: [`test_fsdp_state_dict.py_docs.md`](./test_fsdp_state_dict.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FSDPContainer`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`Model`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`TestDummyModel`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`TestFSDPStateDict`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`TestFSDPStateDict4GPUs`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)

### Functions

- **`__init__`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_broadcast_state_dict`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_compare_models`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_create_module`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_dist_train`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_get_multibuffer_nested_model`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_get_non_fsdp_root_module`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_get_simple_model`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_get_simple_nested_model`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_get_state_dict_mgr`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_initialize_model`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_load_state_dict`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_state_compare`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_state_dict`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_test_state_dict_save_load_flow`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_validate_state_dict_contents`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`apply_ac_to_linears`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`forward`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`get_input`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_basic_save_and_load_state_dict`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_buffers_save_and_load_state_dict`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_fsdp_state_dict_keys`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_fsdp_state_dict_with_activation_checkpoint`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_full_state_dict_missing_unexpected_keys_cleaned`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_local_state_dict_reshard`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_local_state_dict_with_empty_ranks`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_save_and_load_after_forward_state_dict`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_sharded_load_multi_backend_pg`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_shared_module_and_shared_parameter`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_state_dict_load_into_local_module`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_state_dict_rank0_offload_save_load_flow`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_state_dict_save_load_flow`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_state_dict_skip_module`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_state_dict_type`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_state_dict_with_ignored_modules`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_state_dict_with_manual_ac_wrapper`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_state_dict_with_shared_parameters`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_torch_save_load`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_world_size_one`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`test_wrong_state_dict_config`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`world_size`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)

### Imports

- **`Any`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`ChunkShardingSpec`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`DistributedDataParallel`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`FLAT_PARAM`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`FSDP_PREFIX`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`Linear`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`SGD`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`_remote_device`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`contextlib`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`copy`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`deepcopy`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`distributed`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`enable_wrap`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`functools`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`io`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`itertools`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`nullcontext`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`partial`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`sys`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed._shard.sharded_tensor`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed._shard.sharded_tensor.metadata`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed._shard.sharding_spec`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed._state_dict_utils`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed.fsdp._common_utils`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed.fsdp._unshard_param_utils`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.distributed.remote_device`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.nn`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.nn.parallel`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.optim`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)
- **`typing`**: [test_fsdp_state_dict.py_docs.md](./test_fsdp_state_dict.py_docs.md)


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
