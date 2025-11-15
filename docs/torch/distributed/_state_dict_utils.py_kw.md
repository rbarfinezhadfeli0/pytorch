# Keyword Index: `torch/distributed/_state_dict_utils.py`

## File Information

- **Original File**: [torch/distributed/_state_dict_utils.py](../../../torch/distributed/_state_dict_utils.py)
- **Documentation**: [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- **Folder**: `torch/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CompanionMismatch`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_TensorInfo`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)

### Functions

- **`_all_gather_sharded_tensor`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_broadcast_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_broadcast_tensors`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_check_state_dict_similarity`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_copy_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_create_cpu_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_distribute_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_distribute_tensors`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_flatten_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_gather_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_identity_func`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_iterate_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_offload_state_dict_to_cpu`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_set_element`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_traverse_obj`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_traverse_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`_unflatten_state_dict`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`dtensor_func`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`extend_list`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`flat_copy`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`sharded_tensor_func`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`tensor_func`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)

### Imports

- **`Any`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`AsyncCollectiveTensor`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`Callable`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`ShardedTensor`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`collections.abc`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`compute_local_shape_and_global_offset`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`copy`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`distribute_tensor`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`distributed_c10d`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`io`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`math`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`torch`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`torch.cuda._pin_memory_utils`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`torch.distributed`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`torch.distributed._functional_collectives`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`torch.distributed._shard.sharded_tensor`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`torch.distributed.tensor`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`torch.distributed.tensor._utils`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`torch.nn.functional`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`typing`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)
- **`weakref`**: [_state_dict_utils.py_docs.md](./_state_dict_utils.py_docs.md)


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
