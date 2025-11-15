# Keyword Index: `test/distributed/_composable/fsdp/test_fully_shard_extensions.py`

## File Information

- **Original File**: [test/distributed/_composable/fsdp/test_fully_shard_extensions.py](../../../../../test/distributed/_composable/fsdp/test_fully_shard_extensions.py)
- **Documentation**: [`test_fully_shard_extensions.py_docs.md`](./test_fully_shard_extensions.py_docs.md)
- **Folder**: `test/distributed/_composable/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BFloat16AllGatherTensor`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`TestFullyShardAllGatherExtensionsCommon`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`TestFullyShardAllGatherExtensionsMultiProcess`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`TestFullyShardAllGatherExtensionsMultiThread`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`local_param`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)

### Functions

- **`__init__`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`__new__`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`__repr__`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`__tensor_flatten__`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`__tensor_unflatten__`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`__torch_dispatch__`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`_init_two_tensor_mlp`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`_patch_two_tensor_fsdp_all_gather`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`_test_all_gather_extensions_end_to_end`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`_test_all_gather_extensions_train_parity`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`device`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`fsdp_post_all_gather`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`fsdp_pre_all_gather`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`test_all_gather_extension_hsdp_mesh`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`test_all_gather_extension_outer_size_stride`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`test_all_gather_extensions_end_to_end`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`test_all_gather_extensions_monkey_patch`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`test_all_gather_extensions_train_parity`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`two_tensor_fsdp_post_all_gather`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`two_tensor_fsdp_pre_all_gather_v1`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`two_tensor_fsdp_pre_all_gather_v2`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`unwrap`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`world_size`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)

### Imports

- **`Any`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`DeviceMesh`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`TwoTensor`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`_unsafe_preserve_version_counter`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`contextlib`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`copy`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`fully_shard`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`functools`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`math`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`run_tests`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`threading`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.autograd.grad_mode`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.distributed`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.nn`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.testing._internal.two_tensor`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`torch.utils._pytree`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)
- **`typing`**: [test_fully_shard_extensions.py_docs.md](./test_fully_shard_extensions.py_docs.md)


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
