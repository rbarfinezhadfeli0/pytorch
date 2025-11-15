# Keyword Index: `test/distributed/fsdp/test_fsdp_tp_integration.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_tp_integration.py](../../../../test/distributed/fsdp/test_fsdp_tp_integration.py)
- **Documentation**: [`test_fsdp_tp_integration.py_docs.md`](./test_fsdp_tp_integration.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`SimpleModel`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`TestModel`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`TestTPFSDPIntegration`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)

### Functions

- **`__init__`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`_get_grads_as_flattened`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`_get_params_and_sharding_info`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`_get_sub_pgs`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`_sync_tp_grads`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`_test_fsdp_tp_integration`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`assert_local_shard_across_ranks`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`distribute_rmsnorm`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`forward`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`get_non_sharded_param_names`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`get_sharded_param_names`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`prepare_input_fn`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`prepare_output_fn`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`test_fsdp_tp_extension_grad`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`test_fsdp_tp_integration`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`test_fsdp_tp_sync_module_state`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)

### Imports

- **`CommDebugMode`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`FSDPTest`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`Optional`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`OrderedDict`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`collections`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`copy`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`distributed`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`init_device_mesh`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`sys`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.distributed.fsdp.fully_sharded_data_parallel`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.distributed.tensor`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.distributed.tensor.debug`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)
- **`typing`**: [test_fsdp_tp_integration.py_docs.md](./test_fsdp_tp_integration.py_docs.md)


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
