# Keyword Index: `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`

## File Information

- **Original File**: [torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py](../../../../../torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py)
- **Documentation**: [`_fsdp_collectives.py_docs.md`](./_fsdp_collectives.py_docs.md)
- **Folder**: `torch/distributed/fsdp/_fully_shard`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AllGatherResult`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`DefaultAllGather`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`DefaultAllocMixin`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`DefaultReduceScatter`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`ProcessGroupAllocAllGather`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`ProcessGroupAllocMixin`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`ProcessGroupAllocReduceScatter`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)

### Functions

- **`__call__`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`__init__`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`_div_if_needed`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`_get_all_gather_input_metadatas`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`_get_gradient_divide_factors`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`_get_param_all_gather_inputs`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`all_gather_copy_in_cuda`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`all_gather_copy_in_meta`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`allocate`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`chunk_cat`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`foreach_all_gather`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`foreach_all_gather_copy_out`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`foreach_reduce`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`foreach_reduce_scatter_copy_in`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`split_with_sizes_copy`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`use_foreach_copy`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)

### Imports

- **`._fsdp_api`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`._fsdp_common`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`._fsdp_param`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`AllGather`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`Any`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`Callable`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`DTensor`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`FSDPParam`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`ReduceOp`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`_ReduceOp`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`_get_device_handle`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`chain`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`collections.abc`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`itertools`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`math`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`torch`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`torch.distributed`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`torch.distributed.device_mesh`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_api`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`torch.distributed.tensor`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)
- **`typing`**: [_fsdp_collectives.py_docs.md](./_fsdp_collectives.py_docs.md)


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
