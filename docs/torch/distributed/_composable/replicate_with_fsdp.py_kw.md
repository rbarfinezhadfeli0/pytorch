# Keyword Index: `torch/distributed/_composable/replicate_with_fsdp.py`

## File Information

- **Original File**: [torch/distributed/_composable/replicate_with_fsdp.py](../../../../torch/distributed/_composable/replicate_with_fsdp.py)
- **Documentation**: [`replicate_with_fsdp.py_docs.md`](./replicate_with_fsdp.py_docs.md)
- **Folder**: `torch/distributed/_composable`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ReplicateModule`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_ReplicateState`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_ReplicateStateContext`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`and`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`for`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`itself`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)

### Functions

- **`__init__`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`__new__`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_adjust_managed_modules`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_get_managed_modules`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_get_module_replicate_state`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_ignore_module`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_lazy_init`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`dfs`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`init`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`is_composable_with_replicate`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`replicate`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`replicate_impl`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`replicate_mesh`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)

### Imports

- **`.contract`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`DeviceMesh`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`FSDPParamGroup`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`Optional`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`__future__`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_get_device_handle`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_get_module_state`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_get_registry`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`_get_root_modules`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`annotations`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`logging`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed._composable_state`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.device_mesh`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_api`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_common`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_init`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_param_group`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_state`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fully_shard`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.tensor`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.distributed.utils`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`torch.nn`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)
- **`typing`**: [replicate_with_fsdp.py_docs.md](./replicate_with_fsdp.py_docs.md)


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
