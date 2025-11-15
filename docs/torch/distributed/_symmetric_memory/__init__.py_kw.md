# Keyword Index: `torch/distributed/_symmetric_memory/__init__.py`

## File Information

- **Original File**: [torch/distributed/_symmetric_memory/__init__.py](../../../../torch/distributed/_symmetric_memory/__init__.py)
- **Documentation**: [`__init__.py_docs.md`](./__init__.py_docs.md)
- **Folder**: `torch/distributed/_symmetric_memory`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Work`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_ScaleMode`**: [__init__.py_docs.md](./__init__.py_docs.md)

### Functions

- **`__init__`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_all_to_all_vdev_2d_meta`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_all_to_all_vdev_2d_offset_meta`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_check_and_verify_fp8_all_gather_scale_mode`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_chunk_producer`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_all_gather_matmul`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_all_gather_matmul_fallback`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_all_gather_matmul_impl`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_all_gather_matmul_last_gather_dim_impl`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_all_gather_matmul_native`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_all_gather_scaled_matmul`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_all_gather_scaled_matmul_fallback`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_matmul_reduce_scatter`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_matmul_reduce_scatter_fallback`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_matmul_reduce_scatter_impl`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_scaled_matmul_reduce_scatter`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_scaled_matmul_reduce_scatter_fallback`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_fused_scaled_matmul_reduce_scatter_impl`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_get_backend_stream`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_get_remote_tensors_default`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_get_remote_tensors_meta`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_low_contention_all_gather`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_low_contention_all_gather_meta`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_low_contention_reduce_scatter`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_low_contention_reduce_scatter_meta`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_low_contention_reduce_scatter_with_symm_mem_input`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_low_contention_reduce_scatter_with_workspace`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_maybe_convert_scalar_types_to_dtypes`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_multimem_all_gather_matmul`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_pipelined_all_gather_and_consume`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_pipelined_all_gather_and_consume_last_dim`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_pipelined_multi_all_gather_and_consume`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_pipelined_produce_and_all2all`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_should_use_fused_all_gather_matmul_native`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_should_use_multimem_all_gather_matmul`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_test_mode`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`adapter`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`chunk_producer`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`copy_shard`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`default_consumer`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`empty`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`enable_symm_mem_for_group`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`get_backend`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`get_mempool_allocator`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`get_p2p_buf`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`get_p2p_bufs`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`get_symm_mem_workspace`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`is_nvshmem_available`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`is_symm_mem_enabled_for_group`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`make_contiguous_for_perm`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`rendezvous`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`restride_A_for_fused_matmul_reduce_scatter`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`restride_A_shard_for_fused_all_gather_matmul`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`row_wise_replicated_consumer`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`row_wise_sharded_consumer`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`scaled_matmul`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`set_backend`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`unflatten`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`wait`**: [__init__.py_docs.md](./__init__.py_docs.md)

### Imports

- **`Any`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`Callable`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`DeviceType`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`Enum`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`ProcessGroup`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`Sequence`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_SymmetricMemory`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`__future__`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_device`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`_is_nvshmem_available`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`annotations`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`collections.abc`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`contextlib`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`contextmanager`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`datetime`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`enum`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`functools`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`math`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`os`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`overload`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`partial`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`socket`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`timedelta`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`torch`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`torch._C._autograd`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`torch._C._distributed_c10d`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`torch.distributed._functional_collectives`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`torch.types`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`typing`**: [__init__.py_docs.md](./__init__.py_docs.md)
- **`uuid`**: [__init__.py_docs.md](./__init__.py_docs.md)


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
