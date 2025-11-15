# Keyword Index: `test/distributed/_composable/fsdp/test_fully_shard_state_dict.py`

## File Information

- **Original File**: [test/distributed/_composable/fsdp/test_fully_shard_state_dict.py](../../../../../test/distributed/_composable/fsdp/test_fully_shard_state_dict.py)
- **Documentation**: [`test_fully_shard_state_dict.py_docs.md`](./test_fully_shard_state_dict.py_docs.md)
- **Folder**: `test/distributed/_composable/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestFullyShardStateDictMultiProcess`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`TestFullyShardStateDictMultiThread`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)

### Functions

- **`_shard_placement_fn`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`_test_cached_state_dict`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`_test_dp_state_dict_cpu_offload`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`_test_dp_state_dict_save_load`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`_test_dp_tp_state_dict_save_load`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`_test_hsdp_tp_state_dict_save_load`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`_test_state_dict_save_load`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`test_2d_state_dict_correctness`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`test_cached_state_dict`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`test_dp_state_dict_cpu_offload`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`test_dp_state_dict_save_load`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`test_dp_tp_state_dict_save_load`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`test_hsdp_tp_state_dict_save_load`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`test_rank0_offload_full_state_dict`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`world_size`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)

### Imports

- **`CPUOffloadPolicy`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`DeviceMesh`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`Optional`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`contextlib`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`copy`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`distribute_tensor`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`functools`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`nullcontext`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`run_tests`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.distributed.tensor`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.nn`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)
- **`typing`**: [test_fully_shard_state_dict.py_docs.md](./test_fully_shard_state_dict.py_docs.md)


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
