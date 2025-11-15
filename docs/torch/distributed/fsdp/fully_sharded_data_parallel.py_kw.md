# Keyword Index: `torch/distributed/fsdp/fully_sharded_data_parallel.py`

## File Information

- **Original File**: [torch/distributed/fsdp/fully_sharded_data_parallel.py](../../../../torch/distributed/fsdp/fully_sharded_data_parallel.py)
- **Documentation**: [`fully_sharded_data_parallel.py_docs.md`](./fully_sharded_data_parallel.py_docs.md)
- **Folder**: `torch/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FullyShardedDataParallel`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`OptimStateKeyType`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`UnshardHandle`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)

### Functions

- **`__getattr__`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`__getitem__`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`__init__`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_apply`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_assert_state`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_deregister_orig_params_ctx`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_flat_param`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_get_fqn_to_param`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_get_grad_norm`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_get_param_to_fqn`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_has_params`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_is_using_optim_input`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_low_precision_hook_enabled`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_mixed_precision_enabled_for_buffers`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_optim_state_dict_impl`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_optim_state_dict_to_load_impl`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_reset_lazy_init`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_unshard`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_use_training_state`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_wait_unshard_streams_on_current_stream`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_warn_legacy_optim_state_dict`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_warn_optim_input`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`apply`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`check_is_root`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`clip_grad_norm_`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`custom_auto_wrap_policy`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`flatten_sharded_optim_state_dict`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`forward`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`fsdp_modules`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`full_optim_state_dict`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`get_state_dict_type`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`module`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`my_init_fn`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`named_buffers`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`named_parameters`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`no_sync`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`optim_state_dict`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`optim_state_dict_to_load`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`register_comm_hook`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`rekey_optim_state_dict`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`scatter_full_optim_state_dict`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`set_state_dict_type`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`shard_full_optim_state_dict`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`sharded_optim_state_dict`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`state_dict_type`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`summon_full_params`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`wait`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)

### Imports

- **`._flat_param`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`._optim_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`._state_dict_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`._unshard_param_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`.wrap`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`Any`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`Callable`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`CustomPolicy`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`DeviceMesh`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`FlatParameter`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`FullOptimStateDictConfig`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`FullStateDictConfig`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`FullyShardedDataParallel`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`LOW_PRECISION_HOOKS`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`StateDictType`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_annotate_modules_for_dynamo`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_auto_wrap`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_p_assert`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`_register_all_state_dict_hooks`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`auto`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`collections.abc`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`contextlib`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`contextmanager`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`copy`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`enum`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`functools`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`math`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.algorithms._comm_hooks`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.fsdp`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.fsdp._common_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.fsdp._dynamo_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.fsdp._init_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.fsdp._runtime_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.fsdp._traversal_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.fsdp._wrap_utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.fsdp.api`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.tensor`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.distributed.utils`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`torch.nn`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`traceback`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`typing`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)
- **`warnings`**: [fully_sharded_data_parallel.py_docs.md](./fully_sharded_data_parallel.py_docs.md)


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
