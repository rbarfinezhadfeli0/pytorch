# Keyword Index: `torch/sparse/_triton_ops.py`

## File Information

- **Original File**: [torch/sparse/_triton_ops.py](../../../torch/sparse/_triton_ops.py)
- **Documentation**: [`_triton_ops.py_docs.md`](./_triton_ops.py_docs.md)
- **Folder**: `torch/sparse`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TensorAsKey`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)

### Functions

- **`__eq__`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`__hash__`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`__init__`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_bsr_scatter_mm_indices_data`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_bsr_softmax_kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_bsr_strided_addmm_kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_bsr_strided_dense_rowspace_kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_int_bsr_dense_addmm`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_run_sampled_addmm_kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_sampled_addmm_kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_scaled_dot_product_attention`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_scatter_mm2`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_scatter_mm2_kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_scatter_mm6`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`_scatter_mm6_kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`as1Dbatch`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`batch_broadcast_and_squash`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`broadcast_batch_dims`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`broadcast_batch_dims_bsr`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`bsr_dense_addmm`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`bsr_dense_addmm_meta`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`bsr_dense_mm`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`bsr_scatter_mm`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`bsr_scatter_mm_indices_data`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`bsr_softmax`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`check`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`check_blocksize`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`check_bsr_layout`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`check_device`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`check_dtype`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`check_mm_compatible_shapes`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`generate_grid_points`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`generate_sliced_tensors`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`get_tensor_key`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`grid`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`grid_partitioner`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`is_compatible_blocksize`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`is_power_of_two`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`launch_kernel`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`make_triton_contiguous`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`multidim_slicer`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`obj`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`prepare_inputs`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`ptr_stride_extractor`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`sampled_addmm`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`scatter_mm`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`scatter_mm_meta`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`slicer`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`tile_to_blocksize`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`valid_grid_dim`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)

### Imports

- **`._triton_ops_meta`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`functools`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`get_meta`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`has_triton`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`itertools`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`lru_cache`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`math`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`os`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`torch`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`torch._dynamo.utils`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`torch.utils._triton`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`triton`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`triton.language`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`warn_once`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)
- **`weakref`**: [_triton_ops.py_docs.md](./_triton_ops.py_docs.md)


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
