# Keyword Index: `test/distributed/fsdp/test_fsdp_sharded_grad_scaler.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_sharded_grad_scaler.py](../../../../test/distributed/fsdp/test_fsdp_sharded_grad_scaler.py)
- **Documentation**: [`test_fsdp_sharded_grad_scaler.py_docs.md`](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestShardGradScaler`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`TestShardedGradScalerParityWithDDP`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)

### Functions

- **`_build_model_and_optim`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`_get_init_modes_for_test`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`_test_sharded_grad_scaler_found_inf`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`test_fsdp_ddp_parity_with_grad_scaler`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`test_grad_scaling`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`test_inf_gradients_skip_optim_step`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`test_scaling_unscaling_sparse`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`test_sharded_grad_scaler_found_inf`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)

### Imports

- **`CPUOffload`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`DistributedDataParallel`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`ModuleWrapPolicy`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`Optional`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`ShardedGradScaler`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`TransformerDecoderLayer`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`amp_definitely_not_available`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`copy`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`distributed`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`functools`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`itertools`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`sys`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.cuda.amp.common`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.distributed.fsdp.fully_sharded_data_parallel`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.distributed.fsdp.sharded_grad_scaler`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.nn`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.nn.parallel.distributed`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`typing`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)
- **`unittest`**: [test_fsdp_sharded_grad_scaler.py_docs.md](./test_fsdp_sharded_grad_scaler.py_docs.md)


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
