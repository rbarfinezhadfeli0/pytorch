# Keyword Index: `test/distributed/fsdp/test_hsdp_dtensor_state_dict.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_hsdp_dtensor_state_dict.py](../../../../test/distributed/fsdp/test_hsdp_dtensor_state_dict.py)
- **Documentation**: [`test_hsdp_dtensor_state_dict.py_docs.md`](./test_hsdp_dtensor_state_dict.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DenseModel`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`FakeMPModel`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`TestHSDPWithDeviceMeshAndDTensor`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)

### Functions

- **`__init__`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`_create_model`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`forward`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`get_input`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`test_dtensor_sharded_model_load_state_dict`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`test_dtensor_sharded_optim_load_state_dict`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`test_dtensor_sharded_tensor_state_dict_identical`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`test_hsdp_init_with_device_mesh`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`test_root_module_is_not_FSDP`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)

### Imports

- **`DTensor`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`FullyShardedDataParallel`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`ShardedTensor`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`copy`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`deepcopy`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`get_devtype`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`init_device_mesh`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`instantiate_device_type_tests`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`io`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`parametrize`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.distributed`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.distributed._shard.sharded_tensor`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.distributed.fsdp`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.distributed.fsdp.api`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.distributed.tensor`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.nn`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.testing._internal.common_device_type`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_hsdp_dtensor_state_dict.py_docs.md](./test_hsdp_dtensor_state_dict.py_docs.md)


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
