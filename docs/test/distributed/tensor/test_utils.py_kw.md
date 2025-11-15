# Keyword Index: `test/distributed/tensor/test_utils.py`

## File Information

- **Original File**: [test/distributed/tensor/test_utils.py](../../../../test/distributed/tensor/test_utils.py)
- **Documentation**: [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- **Folder**: `test/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FakePlacement`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`LocalTensorTestBase`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`LocalTest`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`MockDeviceMesh`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`Test2DStridedLocalShard`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`TestExplicitRedistribute`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`TestStridedSharding`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`Test_StridedShard_with_shard_order`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`UtilSingleDeviceTest`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`UtilTest`**: [test_utils.py_docs.md](./test_utils.py_docs.md)

### Functions

- **`_compute_start_end_offsets`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`assertEqual`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`build_device_mesh`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`setUp`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`size`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`tearDown`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_1d_mesh_strided_sharding`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_2d_mesh_2d_tensor_strided_sharding`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_2d_mesh_strided_sharding`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_2d_mesh_uneven_strided_shard`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_StridedShard_not_convertible_to_shard_order`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_StridedShard_to_shard_order`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_global_tensor_info_non_shard_placements`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_global_tensor_info_shard_placement`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_global_tensor_info_unsupported_placement`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_global_tensor_shape_1D`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_global_tensor_shape_1D_invalid_shape`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_global_tensor_shape_failure_2D`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_local_shape_and_global_offset_1D`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_local_shape_and_global_offset_2D`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_local_shape_and_global_offset_uneven`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_compute_tensor_info`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_explicit_matmul`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_explicit_order_placements`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_fsdp1_tp_2d_dtensor_local_shards_and_offsets`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_fsdp2_tp_2d_dtensor_local_shards_and_offsets`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_fsdp_tp_meta_compute`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_hsdp_tp_meta_compute`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_strided_sharding_assumption_in_meta_compute`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`test_uneven_fsdp_tp_meta_compute`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`world_size`**: [test_utils.py_docs.md](./test_utils.py_docs.md)

### Imports

- **`Any`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`CommDebugMode`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`DTensorSpec`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`DeviceMesh`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`FakeStore`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`contextlib`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`init_device_mesh`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`itertools`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`nullcontext`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`random`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`run_tests`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.distributed`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.distributed._local_tensor`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.distributed.tensor`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.distributed.tensor._utils`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.distributed.tensor.debug`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.distributed.tensor.placement_types`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_utils.py_docs.md](./test_utils.py_docs.md)
- **`typing`**: [test_utils.py_docs.md](./test_utils.py_docs.md)


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
