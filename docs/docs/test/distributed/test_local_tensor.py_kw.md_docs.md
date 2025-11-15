# Documentation: `docs/test/distributed/test_local_tensor.py_kw.md`

## File Metadata

- **Path**: `docs/test/distributed/test_local_tensor.py_kw.md`
- **Size**: 5,233 bytes (5.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Keyword Index: `test/distributed/test_local_tensor.py`

## File Information

- **Original File**: [test/distributed/test_local_tensor.py](../../../test/distributed/test_local_tensor.py)
- **Documentation**: [`test_local_tensor.py_docs.md`](./test_local_tensor.py_docs.md)
- **Folder**: `test/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`LocalTensorTestBase`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`TestLocalRunner`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`TestLocalTensorWorld2`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`TestLocalTensorWorld3`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`TestLocalTensorWorld4`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`TestLocalTensorWorld8`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`style`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)

### Functions

- **`_get_pp_peer`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`_run_dp_pp`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`assertEqual`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`build_device_mesh`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`setUp`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`tearDown`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_all_gather_collective`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_all_reduce_collective`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_basic_arithmetic_operations`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_broadcast_collective`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_collective_reduction_operations`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_collectives_within_local_tensor_mode`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_dp_pp`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_dtensor_addmm`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_dtensor_cat`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_empty_local_tensors`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_even_sharding_mean_is_partial`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_local_tensor_creation_fails_with_grad_tensors`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_local_tensor_dtype_consistency`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_local_tensor_mode`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_mixed_operations_with_regular_tensors`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_scalar_mul_reduction_bug`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_uneven_sharding_mean_bug`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`test_uneven_sharding_prod`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`world_size`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)

### Imports

- **`contextlib`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`local_p2p_op`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`nullcontext`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`reduce_local_int`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`run_tests`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`torch`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`torch.distributed`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`torch.distributed._local_tensor`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`torch.distributed._local_tensor._c10d`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`torch.distributed.tensor`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_local_tensor.py_docs.md](./test_local_tensor.py_docs.md)


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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/test_local_tensor.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed`):

- [`test_run.py_kw.md_docs.md`](./test_run.py_kw.md_docs.md)
- [`test_inductor_collectives.py_docs.md_docs.md`](./test_inductor_collectives.py_docs.md_docs.md)
- [`test_control_collectives.py_kw.md_docs.md`](./test_control_collectives.py_kw.md_docs.md)
- [`test_c10d_gloo.py_docs.md_docs.md`](./test_c10d_gloo.py_docs.md_docs.md)
- [`test_collective_utils.py_kw.md_docs.md`](./test_collective_utils.py_kw.md_docs.md)
- [`test_data_parallel.py_kw.md_docs.md`](./test_data_parallel.py_kw.md_docs.md)
- [`test_overlap_bucketing_unit.py_kw.md_docs.md`](./test_overlap_bucketing_unit.py_kw.md_docs.md)
- [`test_c10d_nccl.py_kw.md_docs.md`](./test_c10d_nccl.py_kw.md_docs.md)
- [`test_multi_threaded_pg.py_docs.md_docs.md`](./test_multi_threaded_pg.py_docs.md_docs.md)
- [`argparse_util_test.py_kw.md_docs.md`](./argparse_util_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_local_tensor.py_kw.md_docs.md`
- **Keyword Index**: `test_local_tensor.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
