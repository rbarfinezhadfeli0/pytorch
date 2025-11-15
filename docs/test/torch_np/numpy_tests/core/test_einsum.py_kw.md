# Keyword Index: `test/torch_np/numpy_tests/core/test_einsum.py`

## File Information

- **Original File**: [test/torch_np/numpy_tests/core/test_einsum.py](../../../../../test/torch_np/numpy_tests/core/test_einsum.py)
- **Documentation**: [`test_einsum.py_docs.md`](./test_einsum.py_docs.md)
- **Folder**: `test/torch_np/numpy_tests/core`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestEinsum`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`TestEinsumPath`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`TestMisc`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)

### Functions

- **`assert_path_equal`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`build_operands`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`check_einsum_sums`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`optimize_compare`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_broadcasting_dot_cases`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_collapse`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_combined_views_mapping`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_complex`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_different_paths`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_edge_cases`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_edge_paths`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_all_contig_non_contig_output`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_broadcast`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_errors`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_failed_on_p9_and_s390x`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_fixed_collapsingbug`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_fixedstridebug`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_misc`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_cfloat128`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_cfloat64`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_float16`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_float32`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_float64`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_int16`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_int32`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_int64`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_int8`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_sums_uint8`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_einsum_views`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_expand`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_hadamard_like_products`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_index_transformations`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_inner_product`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_long_paths`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_memory_contraints`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_out_is_res`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_output_order`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_overlap`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_path_type_input`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_path_type_input_internal_trace`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_path_type_input_invalid`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_random_cases`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_small_boolean_arrays`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_spaces`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`test_subscript_range`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)

### Imports

- **`expectedFailure`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`functools`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`itertools`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`pytest`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`raises`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`torch._numpy`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`torch._numpy.testing`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)
- **`unittest`**: [test_einsum.py_docs.md](./test_einsum.py_docs.md)


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
