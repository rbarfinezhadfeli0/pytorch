# Keyword Index: `torch/distributed/fsdp/_fully_shard/_fsdp_param.py`

## File Information

- **Original File**: [torch/distributed/fsdp/_fully_shard/_fsdp_param.py](../../../../../torch/distributed/fsdp/_fully_shard/_fsdp_param.py)
- **Documentation**: [`_fsdp_param.py_docs.md`](./_fsdp_param.py_docs.md)
- **Folder**: `torch/distributed/fsdp/_fully_shard`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FSDPParam`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`ShardedState`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`aliasing`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`class`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`if`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`manages`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)

### Functions

- **`__init__`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`__repr__`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_assert_in_states`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_get_grad_inner_tensor`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_init_extensions`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_init_sharded_param`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_init_sharded_post_forward_param_metadata`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_setattr_on_modules`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_sharded_local_tensor`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_unflatten_all_gather_outputs`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`accumulate_unsharded_grad_if_needed`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`all_gather_inputs`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`alloc_all_gather_outputs`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`alloc_storage`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`clear`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`copy_`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`copy__functionalize`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`free_storage`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`free_unsharded_param`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`init_all_gather_outputs`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`init_dtype_attrs`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`init_unsharded_param`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`reset_sharded_param`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`set_requires_grad_if_needed`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`shard_mesh`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`shard_mesh_from_root`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`to_accumulated_grad_if_needed`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`to_sharded`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`to_sharded_dtensor`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`to_sharded_post_forward`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`to_sharded_post_forward_dtensor`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`to_unsharded`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`unsafe_setattr_param`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`unsharded_accumulated_grad_data`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`unsharded_grad_data`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`unsharded_param`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)

### Imports

- **`._fsdp_api`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`._fsdp_common`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`Any`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`AsyncCollectiveTensor`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`CPUOffloadPolicy`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`Callable`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`DDPMeshInfo`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`DTensor`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`DTensorSpec`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`DeviceMesh`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`_StridedShard`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`auto`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`collections.abc`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`dataclass`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`dataclasses`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`enum`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`inspect`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`itertools`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`make_contiguous_strides_for`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch._prims_common`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch.distributed._functional_collectives`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch.distributed.device_mesh`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_common`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch.distributed.tensor`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch.distributed.tensor.placement_types`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`torch.nn`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)
- **`typing`**: [_fsdp_param.py_docs.md](./_fsdp_param.py_docs.md)


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
