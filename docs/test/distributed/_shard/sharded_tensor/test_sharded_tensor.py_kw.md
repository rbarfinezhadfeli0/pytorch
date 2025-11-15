# Keyword Index: `test/distributed/_shard/sharded_tensor/test_sharded_tensor.py`

## File Information

- **Original File**: [test/distributed/_shard/sharded_tensor/test_sharded_tensor.py](../../../../../test/distributed/_shard/sharded_tensor/test_sharded_tensor.py)
- **Documentation**: [`test_sharded_tensor.py_docs.md`](./test_sharded_tensor.py_docs.md)
- **Folder**: `test/distributed/_shard/sharded_tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DummyNNModule`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestCreateTensorFromParams`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestCreateTensorNoProcessGroupMode`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestLocalTensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestModuleHookApi`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardMetadata`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardParameter`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardTensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardedTensorChunked`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardedTensorCustomOps`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardedTensorEnumerable`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardedTensorFromLocalShards`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardedTensorFromLocalTensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardedTensorMetadata`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`TestShardedTensorSubGroupInit`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)

### Functions

- **`__init__`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`_generate_st_from_chunk_local_tensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`create_tensors`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`forward`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`my_op1`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`my_op2`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`my_sharded_asin`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`my_sharded_linear`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_cleanup`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_collect_local_shard`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_complete_world_size`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_create_shard_with_no_placement`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_create_sharded_tensor_like`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_create_sharded_tensor_with_full`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_create_sharded_tensor_with_ones`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_create_sharded_tensor_with_rand`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_create_sharded_tensor_with_zeros`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_custom_op`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_custom_op_errors`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_custom_op_override`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_empty`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_gather_even`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_gather_uneven`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_grid_sharding`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_and_global_metadata`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_and_global_metadata_invalid_shards`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_and_global_metadata_with_all_zeros`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_and_global_metadata_with_local_view`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_invalid_local_shards`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_invalid_pin_memory`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_invalid_property_cross_ranks`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_invalid_shards_gaps`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_invalid_shards_overlap`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_new_group`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_shards_with_different_glb_size`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_tensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_init_from_local_tensor_errors`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_insufficient_sharding_dims`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_invalid_pg_rpc_ranks`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_invalid_sharding`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_load_state_dict_errors`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_local_shards`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_local_tensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_local_tensor_error`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_multiple_local_shards`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_new_group`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_non_contiguous_local_shards`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_non_rw_sharded_recalc_for_metadata`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_partial_world_size`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_recalc_for_metadata`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_reshard_output`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_serialize_and_deserialize`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_shard_metadata_init`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_shard_parameter`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_shard_parameter_errors`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_shard_tensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_shard_tensor_errors`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_shard_tensor_with_empty_shard`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sharded_tensor_device`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sharded_tensor_metadata`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sharded_tensor_sizes`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sharded_tensor_to_cpu`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sharded_tensor_to_cuda`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sharded_tensor_to_test`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sharding_columns`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_st_base_init_from_local_shards_and_global_metadata`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_state_dict`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_state_dict_new_group`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_state_dict_no_sharded_tensors`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sub_process_group_placement_validation`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_sub_process_group_sharded_tensor_init`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_uneven_shards`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`test_with_rpc_names`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`verify_offsets`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`verify_size`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)

### Imports

- **`_remote_device`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`copy`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`custom_sharding_spec_op`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`distributed_c10d`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`io`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`itertools`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`math`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`pickle`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`sharded_tensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`sys`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed._shard`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed._shard.api`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed._shard.sharded_tensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed._shard.sharded_tensor.api`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed._shard.sharded_tensor.utils`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed._shard.sharding_spec`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed._shard.sharding_spec.api`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.distributed.remote_device`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.testing._internal.distributed._shard.sharded_tensor`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)
- **`torch.testing._internal.distributed._shard.sharded_tensor._test_st_common`**: [test_sharded_tensor.py_docs.md](./test_sharded_tensor.py_docs.md)


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
