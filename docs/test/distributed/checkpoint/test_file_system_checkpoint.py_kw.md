# Keyword Index: `test/distributed/checkpoint/test_file_system_checkpoint.py`

## File Information

- **Original File**: [test/distributed/checkpoint/test_file_system_checkpoint.py](../../../../test/distributed/checkpoint/test_file_system_checkpoint.py)
- **Documentation**: [`test_file_system_checkpoint.py_docs.md`](./test_file_system_checkpoint.py_docs.md)
- **Folder**: `test/distributed/checkpoint`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MyShardedModel3`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`MyTestModule`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`TestDistributedReshardOnLoad`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`TestDistributedStateDictSaveLoad`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`TestDistributedStateDictSaveLoadWithCaching`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`TestDistributedStateDictSaveLoadWithSharedTensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)

### Functions

- **`__init__`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`assert_state_dict_equal`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`get_file_path`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`load_tensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`test_load_rowwise_to_colwise`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`test_load_with_different_shard_plan`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`test_read_write_only_tensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`test_read_write_shard_tensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`test_save_load_bytes`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`test_switch_between_sharded_tensor_to_tensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`world_size`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)

### Imports

- **`DefaultSavePlanner`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`ShardedTensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`ZStandard`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`os`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`sharded_tensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`shutil`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`sys`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`tempfile`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.distributed`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.distributed._shard`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.distributed._shard.sharded_tensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.distributed._shard.sharding_spec`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.distributed.checkpoint`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.distributed.checkpoint._extension`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.distributed.checkpoint.default_planner`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.testing._internal.distributed._shard.sharded_tensor`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.testing._internal.distributed._shard.sharded_tensor._test_st_common`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)
- **`torch.testing._internal.distributed.checkpoint_utils`**: [test_file_system_checkpoint.py_docs.md](./test_file_system_checkpoint.py_docs.md)


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
