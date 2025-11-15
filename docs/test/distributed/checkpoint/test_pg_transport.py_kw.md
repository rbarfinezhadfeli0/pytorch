# Keyword Index: `test/distributed/checkpoint/test_pg_transport.py`

## File Information

- **Original File**: [test/distributed/checkpoint/test_pg_transport.py](../../../../test/distributed/checkpoint/test_pg_transport.py)
- **Documentation**: [`test_pg_transport.py_docs.md`](./test_pg_transport.py_docs.md)
- **Folder**: `test/distributed/checkpoint`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`PgTransportCPU`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`PgTransportGPU`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`SimpleModel`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`TestCastTensor`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`TestPGTransportEdgeCases`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`TestPGTransportMocked`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`TestPrepareStateDict`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`TestPrepareTensor`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)

### Functions

- **`__init__`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`_create_sharded_tensor_state_dict`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`_test_pg_transport`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`_test_pg_transport_with_mixed_content`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`_test_pg_transport_with_sharded_tensor`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`backend_str`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`device`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`device_type`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`forward`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`ring_send_recv_checkpoint`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`setUp`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`side_effect`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_cast_tensor_different_dtypes`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_cast_tensor_with_offset`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_cast_tensor_with_stride`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_pg_transport`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_pg_transport_with_mixed_content`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_pg_transport_with_sharded_tensor`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_prepare_state_dict_basic`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_prepare_state_dict_nested`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_prepare_state_dict_with_non_tensor_values`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_prepare_tensor_basic`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_prepare_tensor_different_shapes`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_prepare_tensor_with_stride`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_recv_checkpoint_basic`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_recv_checkpoint_with_state_dict_callback`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_send_checkpoint_basic`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_send_checkpoint_empty_state_dict`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_send_checkpoint_with_cpu_tensors`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`test_send_checkpoint_with_non_tensor_values`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)

### Imports

- **`DTensor`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`MagicMock`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`Optional`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`_get_default_group`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`datetime`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`init_device_mesh`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`logging`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`timedelta`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.distributed`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.distributed._shard.sharded_tensor`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.distributed.checkpoint._pg_transport`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.distributed.tensor`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.nn`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`torch.utils._pytree`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`tree_flatten_with_path`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`typing`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`unittest`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)
- **`unittest.mock`**: [test_pg_transport.py_docs.md](./test_pg_transport.py_docs.md)


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
