# Keyword Index: `torch/distributed/tensor/placement_types.py`

## File Information

- **Original File**: [torch/distributed/tensor/placement_types.py](../../../../torch/distributed/tensor/placement_types.py)
- **Documentation**: [`placement_types.py_docs.md`](./placement_types.py_docs.md)
- **Folder**: `torch/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MaskPartial`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`Partial`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`Replicate`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`Shard`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_StridedShard`**: [placement_types.py_docs.md](./placement_types.py_docs.md)

### Functions

- **`__eq__`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`__hash__`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`__init__`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`__repr__`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`__str__`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_compute_padding_info`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_get_shard_pad_size`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_local_shard_size`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_local_shard_size_and_offset`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_make_replicate_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_make_shard_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_mask_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_maybe_pad_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_maybe_unpad_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_maybe_unpad_tensor_with_sizes`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_pad_for_new_shard_dim`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_partition_value`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_reduce_shard_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_reduce_shard_value`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_reduce_value`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_replicate_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_replicate_to_shard`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_select_shard`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_shard_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_split_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_to_new_shard_dim`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_to_replicate_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`_unpad_for_new_shard_dim`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`local_shard_size_and_offset`**: [placement_types.py_docs.md](./placement_types.py_docs.md)

### Imports

- **`DeviceMesh`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`MaskBuffer`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`Placement`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`cast`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`dataclass`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`dataclasses`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`maybe_run_for_local_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`torch`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`torch._C`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`torch._C._distributed`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`torch.distributed._functional_collectives`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`torch.distributed._local_tensor`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`torch.distributed.device_mesh`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`torch.distributed.tensor._collective_utils`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`torch.distributed.tensor._ops._mask_buffer`**: [placement_types.py_docs.md](./placement_types.py_docs.md)
- **`typing`**: [placement_types.py_docs.md](./placement_types.py_docs.md)


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
