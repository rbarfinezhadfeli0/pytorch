# Keyword Index: `torch/testing/_internal/distributed/_tensor/common_dtensor.py`

## File Information

- **Original File**: [torch/testing/_internal/distributed/_tensor/common_dtensor.py](../../../../../../torch/testing/_internal/distributed/_tensor/common_dtensor.py)
- **Documentation**: [`common_dtensor.py_docs.md`](./common_dtensor.py_docs.md)
- **Folder**: `torch/testing/_internal/distributed/_tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Attention`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`DTensorContinuousTestBase`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`DTensorConverter`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`DTensorOpTestBase`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`DTensorTestBase`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`FeedForward`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`LocalDTensorTestBase`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`MLPModule`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`MLPStacked`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`RMSNormPython`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`Transformer`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`TransformerBlock`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`class`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`for`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`from`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)

### Functions

- **`__init__`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`__iter__`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`__next__`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`_convert_shard_order_dict_to_ShardOrder`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`_get_local_tensor_mode`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`_handle_test_skip`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`_norm`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`_spawn_processes`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`_split_list`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`_test_op`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`_test_op_on_dtensor`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`backend`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`backend_str`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`build_device_mesh`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`compositions`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`create_local_tensor_test_class`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`decorator`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`destroy_pg`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`device_type`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`forward`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`gen_sharding_choices_for_arg`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`generate_shard_orders`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`init_manual_seed_for_rank`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`init_pg`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`is_local_tensor_enabled`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`is_supported_tensor`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`join_or_run`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`make_full_tensor`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`make_wrapped`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`map_local_for_rank`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`map_local_tensor_for_rank`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`parallelize`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`patched_distribute_tensor`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`rank`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`redistribute`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`reduce_local_int`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`reset_parameters`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`run_subtests`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`run_test`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`setUp`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`shard_order_to_placement`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`skip_unless_torch_gpu`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`successful`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`tearDown`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`test_some_method`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`to_dist_tensor`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`with_comms`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`world_size`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`wrapped`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`wrapper`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)

### Imports

- **`Any`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`Callable`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`ShardOrderEntry`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`collections.abc`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`contextlib`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`copy`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`dataclass`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`dataclasses`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`functools`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`itertools`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`partial`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`redistribute_local_tensor`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`sys`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.distributed`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.distributed._local_tensor`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.distributed.tensor`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.distributed.tensor._redistribute`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.nn`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.nn.functional`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.testing._internal.common_utils`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`torch.utils._pytree`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`tree_flatten`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`types`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)
- **`typing`**: [common_dtensor.py_docs.md](./common_dtensor.py_docs.md)


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
