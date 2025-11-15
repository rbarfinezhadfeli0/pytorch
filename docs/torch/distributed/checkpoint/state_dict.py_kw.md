# Keyword Index: `torch/distributed/checkpoint/state_dict.py`

## File Information

- **Original File**: [torch/distributed/checkpoint/state_dict.py](../../../../torch/distributed/checkpoint/state_dict.py)
- **Documentation**: [`state_dict.py_docs.md`](./state_dict.py_docs.md)
- **Folder**: `torch/distributed/checkpoint`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`_EXTRA_STATE`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`class`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`specifies`**: [state_dict.py_docs.md](./state_dict.py_docs.md)

### Functions

- **`_device`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_flatten_optim_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_flatten_state_nested_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_gc_context`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_get_fqns`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_get_model_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_get_optim_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_init_optim_state`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_iterate_valid_model_state`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_load_model_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_load_optim_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_maybe_full_or_cpu_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_patch_model_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_patch_optimizer_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_raise_if_type_not_supported`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_reconstruct_nested_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_split_optim_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_state_dict_fn`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_unflatten_model_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_unflatten_optim_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_verify_options`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_verify_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`fsdp_state_dict_type_without_warning`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`get_model_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`get_optimizer_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`get_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`load_state_dict_call`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`recurse`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`set_model_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`set_optimizer_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`set_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`state_dict_call`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`verify`**: [state_dict.py_docs.md](./state_dict.py_docs.md)

### Imports

- **`Any`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`Callable`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`DTensor`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`DistributedDataParallel`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`FullyShardedDataParallel`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`ShardedTensor`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`_IncompatibleKeys`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`asdict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`chain`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`collections.abc`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`contextlib`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`dataclasses`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`functools`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`gc`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`get_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`itertools`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`patch_model_state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.distributed`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.distributed._shard.sharded_tensor`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.distributed._state_dict_utils`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.distributed.checkpoint.state_dict`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.distributed.fsdp`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.distributed.fsdp._common_utils`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.distributed.tensor`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.nn`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.nn.modules.module`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.nn.parallel`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`torch.utils._pytree`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`tree_map_only`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`typing`**: [state_dict.py_docs.md](./state_dict.py_docs.md)
- **`warnings`**: [state_dict.py_docs.md](./state_dict.py_docs.md)


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
