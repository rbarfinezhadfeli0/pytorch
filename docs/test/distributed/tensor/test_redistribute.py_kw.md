# Keyword Index: `test/distributed/tensor/test_redistribute.py`

## File Information

- **Original File**: [test/distributed/tensor/test_redistribute.py](../../../../test/distributed/tensor/test_redistribute.py)
- **Documentation**: [`test_redistribute.py_docs.md`](./test_redistribute.py_docs.md)
- **Folder**: `test/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DistributeWithDeviceOrderTest`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`MultiDimRedistributeTest`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`RedistributeTest`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)

### Functions

- **`_compute_local_shape`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`_extract_redistribute_trace_from_debug_mode`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`_is_valid_placement`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_generate_shard_orders`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_multi_dim_mesh`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_one_chunk_mesh`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_ordered_distribute_all_combination`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_ordered_redistribute`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_ordered_redistribute_for_special_placement`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_ordered_redistribute_with_partial`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_partial_to_replicate_forward_backward`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_partial_to_shard`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_redistribute_negative_shard_dim`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_redistribute_shard_dim_change`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_redistribute_shard_dim_multi_dim_mesh`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_redistribute_to_partial`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_redistribute_uneven_sharding`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_replicate_to_local_partial_grad`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_replicate_to_partial`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_replicate_to_replicate_forward_backward`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_replicate_to_replicate_forward_backward_datatype_conversion`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_replicate_to_shard_forward_backward`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_shard_dim_alltoall`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_shard_order_same_data_as_strided_shard`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_shard_to_replicate_forward_backward`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`test_shard_to_replicate_forward_backward_datatype_conversion`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`world_size`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)

### Imports

- **`CommDebugMode`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`DebugMode`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`Redistribute`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`ShardOrderEntry`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`_StridedShard`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`contextlib`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`init_device_mesh`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`itertools`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`math`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`re`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`shard_dim_alltoall`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.distributed._local_tensor`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.distributed.tensor`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.distributed.tensor._collective_utils`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.distributed.tensor._redistribute`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.distributed.tensor.debug`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.distributed.tensor.placement_types`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`torch.utils._debug_mode`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)
- **`unittest`**: [test_redistribute.py_docs.md](./test_redistribute.py_docs.md)


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
