# Keyword Index: `torch/testing/_internal/distributed/ddp_under_dist_autograd_test.py`

## File Information

- **Original File**: [torch/testing/_internal/distributed/ddp_under_dist_autograd_test.py](../../../../../torch/testing/_internal/distributed/ddp_under_dist_autograd_test.py)
- **Documentation**: [`ddp_under_dist_autograd_test.py_docs.md`](./ddp_under_dist_autograd_test.py_docs.md)
- **Folder**: `torch/testing/_internal/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CommonDdpComparisonTest`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`CudaDdpComparisonTest`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`DdpComparisonTest`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`DdpMode`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`DdpUnderDistAutogradTest`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`FeatureSet`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`HybridModel`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`RemoteEM`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`RemoteNet`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`Trainer`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)

### Functions

- **`__init__`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`_call_method`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`_do_test`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`_master_process`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`_remote_method`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`_remote_method_async`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`_remote_worker_process`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`_run_test_ddp_comparision`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`_trainer_process`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`destroy_pg`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`do_test_on_master`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`forward`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`getLinear`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`get_remote_grads`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`get_training_examples`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`init_logger`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`remote_worker_name`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`set_shutdown_signal`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_backward_ddp_inside`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_backward_ddp_outside`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_backward_ddp_outside_uneven_inputs`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_backward_no_ddp`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_ddp_comparison`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_ddp_comparison_uneven_inputs`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_ddp_dist_autograd_local_vs_remote`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_ddp_dist_autograd_local_vs_remote_gpu`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`test_ddp_dist_autograd_sparse_grads`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`train_batch`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`trainer_name`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`world_size`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)

### Imports

- **`DistributedDataParallel`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`NamedTuple`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`RemoteModule`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`contextlib`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`dist_init`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`enum`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`logging`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`os`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`rpc`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`threading`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch.distributed`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch.distributed.autograd`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch.distributed.nn`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch.nn`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch.nn.parallel`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch.testing._internal.dist_utils`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`torch.testing._internal.distributed.rpc.rpc_agent_test_fixture`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)
- **`typing`**: [ddp_under_dist_autograd_test.py_docs.md](./ddp_under_dist_autograd_test.py_docs.md)


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
