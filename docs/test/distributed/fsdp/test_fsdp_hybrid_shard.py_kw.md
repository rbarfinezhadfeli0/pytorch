# Keyword Index: `test/distributed/fsdp/test_fsdp_hybrid_shard.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_hybrid_shard.py](../../../../test/distributed/fsdp/test_fsdp_hybrid_shard.py)
- **Documentation**: [`test_fsdp_hybrid_shard.py_docs.md`](./test_fsdp_hybrid_shard.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MyModel`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`ShardingStrategyMode`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`TestFSDPHybridShard`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)

### Functions

- **`__init__`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`_init_fsdp_model`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`_init_hsdp_model`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`_test_fsdp_hybrid_shard_basic_setup`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`_test_fsdp_hybrid_shard_parity`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`forward`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`patch_allreduce`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`patch_reduce_scatter`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`patched_collective`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`process_group`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`test_fsdp_hybrid_shard_basic_setup`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`test_fsdp_hybrid_shard_parity`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`test_hsdp_save_load_state_dict`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`test_hsdp_sync_module_state`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`test_invalid_pg_specification_raises`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`test_raises_manual_wrap_hybrid_shard_when_none_policy`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`world_size`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)

### Imports

- **`Counter`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`ModuleWrapPolicy`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`Optional`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`TransformerDecoderLayer`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`_rank_not_in_group`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`auto`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`collections`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`contextlib`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`enum`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`functools`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`init_device_mesh`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`partial`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`sys`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.distributed`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.distributed.fsdp._init_utils`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.distributed.fsdp._traversal_utils`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.nn`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)
- **`typing`**: [test_fsdp_hybrid_shard.py_docs.md](./test_fsdp_hybrid_shard.py_docs.md)


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
