# Keyword Index: `torch/testing/_internal/distributed/nn/api/remote_module_test.py`

## File Information

- **Original File**: [torch/testing/_internal/distributed/nn/api/remote_module_test.py](../../../../../../../torch/testing/_internal/distributed/nn/api/remote_module_test.py)
- **Documentation**: [`remote_module_test.py_docs.md`](./remote_module_test.py_docs.md)
- **Folder**: `torch/testing/_internal/distributed/nn/api`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BadModule`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`CommonRemoteModuleTest`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`CudaRemoteModuleTest`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`ModuleCreationMode`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`MyModule`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`MyModuleInterface`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`RemoteModuleTest`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`RemoteMyModuleInterface`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`ThreeWorkersRemoteModuleTest`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`nn`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)

### Functions

- **`__init__`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`_create_remote_module_iter`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`create_scripted_module`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`forward`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`forward_async`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`get_remote_training_arg`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`hook`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`remote_device`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`remote_forward`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`remote_forward_async`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`remote_module_attributes`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`run_forward`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`run_forward_async`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_bad_module`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_create_remote_module_from_module_rref`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_forward_async`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_forward_async_script`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_forward_sync`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_forward_sync_script`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_forward_with_kwargs`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_get_module_rref`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_input_moved_to_cuda_device`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_input_moved_to_cuda_device_script`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_invalid_devices`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_remote_module_py_pickle_not_supported`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_remote_module_py_pickle_not_supported_script`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_remote_parameters`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_send_remote_module_over_the_wire`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_send_remote_module_over_the_wire_script_not_supported`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_send_remote_module_with_a_new_attribute_not_pickled_over_the_wire`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_train_eval`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_unsupported_methods`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`test_valid_device`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`world_size`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)

### Imports

- **`Future`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`RemoteModule`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`TemporaryFileName`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`enum`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`nn`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`skip_if_lt_x_gpu`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch._jit_internal`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch.distributed.nn`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch.distributed.nn.api.remote_module`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch.distributed.rpc`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch.testing._internal.common_utils`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch.testing._internal.dist_utils`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)
- **`torch.testing._internal.distributed.rpc.rpc_agent_test_fixture`**: [remote_module_test.py_docs.md](./remote_module_test.py_docs.md)


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
