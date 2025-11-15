# Keyword Index: `test/distributed/_composable/fsdp/test_fully_shard_ignore_params.py`

## File Information

- **Original File**: [test/distributed/_composable/fsdp/test_fully_shard_ignore_params.py](../../../../../test/distributed/_composable/fsdp/test_fully_shard_ignore_params.py)
- **Documentation**: [`test_fully_shard_ignore_params.py_docs.md`](./test_fully_shard_ignore_params.py_docs.md)
- **Folder**: `test/distributed/_composable/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`A`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`B`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`C`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`TestFullyShardIgnoreParams`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`X`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`Y`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)

### Functions

- **`__init__`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_append_prefix`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_discover_ddp_ignored_params`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_discover_fsdp_ignored_params`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_find_all_fsdped_modules`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_find_name_param_mappings`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_generate_model_and_input`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_get_full_tensor`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_modify_ddp_ignored_params`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`_post_order_wrap_fsdp`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`compare_params`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`compare_ref_test_params`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`forward`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`test_ddp_A_fsdp_B_ddp_C`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)

### Imports

- **`DTensor`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`DistributedDataParallel`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`FSDPModule`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`FSDPTest`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`fully_shard`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`implicit_replication`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`init_device_mesh`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`sys`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.distributed`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.distributed._composable.fsdp`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.distributed._composable.fsdp.fully_shard`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.distributed.tensor`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.distributed.tensor.experimental`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.nn`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.nn.parallel`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fully_shard_ignore_params.py_docs.md](./test_fully_shard_ignore_params.py_docs.md)


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
