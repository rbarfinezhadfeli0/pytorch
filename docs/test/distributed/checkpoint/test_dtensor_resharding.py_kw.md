# Keyword Index: `test/distributed/checkpoint/test_dtensor_resharding.py`

## File Information

- **Original File**: [test/distributed/checkpoint/test_dtensor_resharding.py](../../../../test/distributed/checkpoint/test_dtensor_resharding.py)
- **Documentation**: [`test_dtensor_resharding.py_docs.md`](./test_dtensor_resharding.py_docs.md)
- **Folder**: `test/distributed/checkpoint`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CheckpointableDistTensor`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`TestCheckpointableReshard`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`TestDTensorReshardMeshChange`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`TestDTensorReshardPlacementChange`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)

### Functions

- **`__create_chunk_list__`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`__create_write_items__`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`__get_tensor_shard__`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`__init__`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`__new__`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`__repr__`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`__torch_dispatch__`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`test_1d_to_1d_reshard_placement_change`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`test_1d_to_2d_reshard_mesh_change`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`test_2d_to_1d_reshard_mesh_change`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`test_2d_to_2d_reshard_placement_change`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`test_dtensor_checkpoint_resharding_with_empty_shard`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`test_dtensor_checkpoint_with_uneven_shards`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`test_uneven_reshard_with_checkpointable_api`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`test_uneven_reshard_with_dtensor_shards_wrapper_api`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)

### Imports

- **`Any`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`LocalShardsWrapper`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`ZStandard`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`distribute_tensor`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`init_device_mesh`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`logging`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.distributed`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.distributed.checkpoint`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.distributed.checkpoint._extension`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.distributed.checkpoint.metadata`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.distributed.checkpoint.planner`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.distributed.tensor`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.distributed.tensor._shards_wrapper`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`torch.testing._internal.distributed.checkpoint_utils`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)
- **`typing`**: [test_dtensor_resharding.py_docs.md](./test_dtensor_resharding.py_docs.md)


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
