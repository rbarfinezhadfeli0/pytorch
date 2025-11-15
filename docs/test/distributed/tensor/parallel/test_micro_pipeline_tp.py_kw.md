# Keyword Index: `test/distributed/tensor/parallel/test_micro_pipeline_tp.py`

## File Information

- **Original File**: [test/distributed/tensor/parallel/test_micro_pipeline_tp.py](../../../../../test/distributed/tensor/parallel/test_micro_pipeline_tp.py)
- **Documentation**: [`test_micro_pipeline_tp.py_docs.md`](./test_micro_pipeline_tp.py_docs.md)
- **Folder**: `test/distributed/tensor/parallel`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MicroPipelineTP4GPUTest`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`MicroPipelineTPTest`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)

### Functions

- **`_fp8_all_gather`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`_make_post_grad_fx`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`func`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`reshape_mm_reshape`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`setUp`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`tearDown`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_dtensor_seq_par`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_extra_collectives`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_find_all_gather_patterns`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_find_reduce_scatter_patterns`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_fuse_all_gather_matmul`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_fuse_all_gather_scaled_matmul`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_fuse_matmul_reduce_scatter`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_fuse_scaled_matmul_reduce_scatter`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_fuse_scaled_matmul_reduce_scatter_rowwise_scales_reshape_mm_reshape`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`test_get_unexposed_collectives`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)

### Imports

- **`DeviceMesh`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`FakeStore`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`HAS_GPU`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`MLPModule`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`Optional`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`_get_group_size_by_name`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`_test_mode`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`decompositions`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`e4m3_type`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`fresh_cache`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`functorch`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`make_fx`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`remove_noop_ops`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch._inductor.decomposition`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch._inductor.fx_passes.micro_pipeline_tp`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch._inductor.fx_passes.post_grad`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch._inductor.utils`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.distributed`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.distributed._functional_collectives`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.distributed._symmetric_memory`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.distributed.tensor`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.testing._internal.common_device_type`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`typing`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)
- **`unittest`**: [test_micro_pipeline_tp.py_docs.md](./test_micro_pipeline_tp.py_docs.md)


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
