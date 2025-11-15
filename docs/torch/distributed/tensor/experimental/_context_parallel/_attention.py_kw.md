# Keyword Index: `torch/distributed/tensor/experimental/_context_parallel/_attention.py`

## File Information

- **Original File**: [torch/distributed/tensor/experimental/_context_parallel/_attention.py](../../../../../../torch/distributed/tensor/experimental/_context_parallel/_attention.py)
- **Documentation**: [`_attention.py_docs.md`](./_attention.py_docs.md)
- **Folder**: `torch/distributed/tensor/experimental/_context_parallel`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AttentionType`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_AllGatherRotater`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_AllToAllRotater`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_AttentionOp`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_CausalBehavior`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_ContextParallel`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_DispatchMode`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_RingRotater`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_RotateMethod`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_SDPAMerger`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`class`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`from`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`to`**: [_attention.py_docs.md](./_attention.py_docs.md)

### Functions

- **`__call__`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`__init__`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_apply`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_context_parallel_buffers`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_context_parallel_shard`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_create_cp_block_mask`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_create_rotater`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_disable_context_parallel_dispatcher`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_disable_context_parallel_dispatcher_impl`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_disable_cp_dtensor_dispatcher`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_distribute_function`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_enable_context_parallel_dispatcher`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_enable_context_parallel_dispatcher_impl`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_enable_cp_dtensor_dispatcher`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_is_causal_behavior`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_maybe_wait`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_merge_one`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_partial_update`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_restore_function`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_rewrite_mask_mod`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_scaled_dot_product_ring_cudnn_attention`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_scaled_dot_product_ring_cudnn_attention_backward`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_scaled_dot_product_ring_efficient_attention`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_scaled_dot_product_ring_efficient_attention_backward`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_scaled_dot_product_ring_flash_attention`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_scaled_dot_product_ring_flash_attention_backward`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_sdpa_handler`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_templated_ring_attention`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_templated_ring_attention_backward`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`context_parallel`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`context_parallel_unshard`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`exchange_buffers`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`flex_input_fn`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`inner_fn`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`local_q_idx_to_q_idx`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`next_buffer`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`qkv_idx_restore`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`results`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`sdpa_input_fn`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`sdpa_output_fn`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`set_rotate_method`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`step`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`wrapper`**: [_attention.py_docs.md](./_attention.py_docs.md)

### Imports

- **`._cp_custom_ops`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`._load_balancer`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`ABC`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`Any`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`Callable`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`DeviceMesh`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`ParallelStyle`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_DEFAULT_SPARSE_BLOCK_SIZE`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`_create_default_load_balancer`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`abc`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`auto`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`collections.abc`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`contextlib`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`dataclass`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`dataclasses`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`distribute_tensor`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`enum`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`flex_cp_allgather`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`functools`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`itertools`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`logging`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`partial`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.distributed`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.distributed._functional_collectives`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.distributed.device_mesh`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.distributed.tensor`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.nn`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.nn.attention.flex_attention`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.nn.functional`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`torch.utils._pytree`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`tree_flatten`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`types`**: [_attention.py_docs.md](./_attention.py_docs.md)
- **`typing`**: [_attention.py_docs.md](./_attention.py_docs.md)


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
