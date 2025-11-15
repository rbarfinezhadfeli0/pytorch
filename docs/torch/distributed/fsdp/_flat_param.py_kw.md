# Keyword Index: `torch/distributed/fsdp/_flat_param.py`

## File Information

- **Original File**: [torch/distributed/fsdp/_flat_param.py](../../../../torch/distributed/fsdp/_flat_param.py)
- **Documentation**: [`_flat_param.py_docs.md`](./_flat_param.py_docs.md)
- **Folder**: `torch/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FlatParamHandle`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`FlatParamShardMetadata`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`FlatParameter`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`HandleShardingStrategy`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`ParamInfo`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`SharedParamInfo`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_FlatParameterMeta`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_ShardParamInfo`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`docstring`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`for`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`super`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)

### Functions

- **`__init__`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`__instancecheck__`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`__new__`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`__repr__`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_all_gather_flat_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_alloc_padded_unsharded_flat_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_check_low_precision_shard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_check_on_compute_device`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_check_on_cpu`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_check_sharded`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_check_sharded_strategy`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_check_storage_allocated`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_check_storage_freed`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_check_unsharded`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_construct_padding_tensor`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_convert_to_params`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_deregister_orig_params`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_detach_if_needed`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_force_full_precision`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_fqns_in_shard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_free_low_precision_sharded_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_free_unsharded_flat_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_aligned_numel`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_dtype_size`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_flat_param_offsets`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_modules`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_padded_unsharded_flat_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_shard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_shard_metadata`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_sharded_size`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_unflat_views_aligned`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_unflat_views_unaligned`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_get_unpadded_shard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_init_flat_param_and_metadata`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_init_get_unflat_views_fn`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_init_metadata`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_init_param_reduce_dtypes`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_init_setattr_fns`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_init_shard_metadata`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_is_truly_contiguous`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_reset_flat_param_grad_info_if_needed`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_reset_is_grad_none`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_safe_setattr_tensor_or_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_same_storage`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_same_storage_size`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_skipped_use_sharded_views`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_storage_size_allocated`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_unsafe_setattr_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_unsafe_setattr_tensor`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_use_low_precision_shard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_use_sharded_flat_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_use_sharded_grad_views`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_use_sharded_views`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_use_unsharded_flat_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_use_unsharded_grad_views`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_use_unsharded_views`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_uses_param_mixed_precision`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_uses_reduce_mixed_precision`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_validate_tensors_to_flatten`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_warn_skip_writeback_check`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_warn_use_fake_all_gather`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_warn_use_fake_reduce`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_writeback_orig_params`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_writeback_tensor`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`cast_grad_to_param_dtype_if_needed`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`flat_param_to`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`flatten_tensors`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`flatten_tensors_into_flat_param`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`init_flat_param_attributes`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`is_sharded`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`needs_unshard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`param_module_names`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`post_reshard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`post_unshard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`pre_unshard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`prepare_gradient_for_backward`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`prepare_gradient_for_optim`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`reshard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`reshard_grad`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`shard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`shard_metadata`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`sharded_grad`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`shared_param_module_names`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`to_cpu`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`unflatten_as_params`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`unshard`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`unshard_grad`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`uses_sharded_strategy`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)

### Imports

- **`._fsdp_extensions`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`Any`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`Callable`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`DTensor`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`FakeProcessGroup`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`Tensor`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`_ParameterMeta`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`accumulate`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`auto`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`collections.abc`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`contextlib`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`enum`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`functools`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`itertools`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`logging`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`os`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch.distributed`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch.distributed.fsdp._common_utils`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch.distributed.tensor`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch.distributed.utils`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch.nn`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch.nn.functional`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch.nn.parameter`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`typing`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)
- **`warnings`**: [_flat_param.py_docs.md](./_flat_param.py_docs.md)


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
