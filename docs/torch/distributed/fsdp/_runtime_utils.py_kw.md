# Keyword Index: `torch/distributed/fsdp/_runtime_utils.py`

## File Information

- **Original File**: [torch/distributed/fsdp/_runtime_utils.py](../../../../torch/distributed/fsdp/_runtime_utils.py)
- **Documentation**: [`_runtime_utils.py_docs.md`](./_runtime_utils.py_docs.md)
- **Folder**: `torch/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`_PrefetchMode`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)

### Functions

- **`_accumulate_sharded_grad`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_cast_buffers_to_dtype_and_device`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_cast_grad_to_param_dtype`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_catch_all_reshard`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_check_flat_params_on_expected_device`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_check_grad_to_accumulate`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_div_if_needed`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_finalize_params`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_get_buffers_and_dtypes_for_computation`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_get_fsdp_root_states`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_get_fsdp_root_states_with_modules`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_get_handle_to_prefetch`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_get_orig_buffer_dtypes`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_get_reduce_scatter_tensors`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_get_training_state`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_init_streams`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_is_fsdp_root`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_lazy_init`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_low_precision_hook_enabled`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_offload_grad`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_post_backward_final_callback`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_post_backward_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_post_backward_reshard`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_post_backward_reshard_only_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_post_backward_use_sharded_grad_views`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_post_forward`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_post_forward_reshard`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_post_reduce_grad_callback`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_pre_backward_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_pre_forward`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_pre_forward_unshard`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_prefetch_handle`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_reduce_grad`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_reduce_grad_no_shard`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_register_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_register_post_backward_final_callback`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_register_post_backward_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_register_post_backward_reshard_only_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_register_post_forward_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_register_pre_backward_hooks`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_register_pre_forward_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_register_root_pre_forward_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_reset_flat_param_grad_info_if_needed`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_reshard`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_reshard_grads`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_root_cast_forward_input`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_root_pre_forward`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_share_state_and_init_handle_attrs`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_should_free_in_backward`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_unshard`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_unshard_grads`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_wait_for_computation_stream`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)

### Imports

- **`Any`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`BackwardPrefetch`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`Callable`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`HYBRID_SHARDING_STRATEGIES`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`LOW_PRECISION_HOOKS`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`Variable`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`_pytree`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`auto`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`collections.abc`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`enum`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`functools`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`logging`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`register_multi_grad_hook`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.autograd`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.autograd.graph`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.distributed`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.distributed.algorithms._comm_hooks`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.distributed.fsdp._common_utils`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.distributed.fsdp._flat_param`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.distributed.fsdp._init_utils`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.distributed.fsdp._traversal_utils`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.distributed.fsdp.api`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.distributed.utils`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.nn`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.nn.functional`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`torch.utils`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)
- **`typing`**: [_runtime_utils.py_docs.md](./_runtime_utils.py_docs.md)


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
