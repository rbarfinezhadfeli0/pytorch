# Keyword Index: `test/distributed/test_fake_pg.py`

## File Information

- **Original File**: [test/distributed/test_fake_pg.py](../../../test/distributed/test_fake_pg.py)
- **Documentation**: [`test_fake_pg.py_docs.md`](./test_fake_pg.py_docs.md)
- **Folder**: `test/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`SimpleTensorMode`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`TestFakePG`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)

### Functions

- **`__init__`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`__torch_dispatch__`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`allgather_fn`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`tearDown`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_all_reduce`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_allgather`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_alltoall`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_alltoall_base`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_broadcast`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_construct_fsdp`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_error_on_collective`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_fake_pg_tracing`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_fake_process_group_direct_usage_error`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_fake_process_group_proper_usage_dispatch`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_fsdp_fake_e2e`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_fsdp_tp_fake_e2e`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_recv`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_reduce_scatter`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_scatter`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`test_send`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)

### Imports

- **`DeviceMesh`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`FakeProcessGroup`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`FakeStore`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`FileCheck`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`FullyShardedDataParallel`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`HAS_ACCELERATOR`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`MLPModule`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`TorchDispatchMode`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`get_devtype`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`init_device_mesh`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`make_fx`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`run_tests`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`sys`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch._C._distributed_c10d`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.distributed`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.distributed._functional_collectives`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.distributed.tensor`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.nn`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.testing`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`torch.utils._python_dispatch`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)
- **`unittest`**: [test_fake_pg.py_docs.md](./test_fake_pg.py_docs.md)


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
