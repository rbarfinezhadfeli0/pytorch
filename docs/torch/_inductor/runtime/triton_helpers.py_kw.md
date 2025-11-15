# Keyword Index: `torch/_inductor/runtime/triton_helpers.py`

## File Information

- **Original File**: [torch/_inductor/runtime/triton_helpers.py](../../../../torch/_inductor/runtime/triton_helpers.py)
- **Documentation**: [`triton_helpers.py_docs.md`](./triton_helpers.py_docs.md)
- **Folder**: `torch/_inductor/runtime`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_any_combine`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`_bitonic_merge_with_index`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`_compare_and_swap_with_index`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`_prod_accumulate`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`any`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`bucketize_binary_search`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`constexpr_next_power_of_2`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`device_assert_then`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`div_floor_integer`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`exclusive_scan_decoupled_lookback`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`exclusive_scan_decoupled_lookback_64`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`exp`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`frexp`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`get_backend_options`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`get_constexprs`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`if_mask`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`is_floating`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`max2`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`max_with_index`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`maximum`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`maximum_with_index`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`min2`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`min_with_index`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`minimum`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`minimum_with_index`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`online_softmax_combine`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`online_softmax_reduce`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`pack_value_flag`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`prod`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`promote_to_tensor`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`randint64`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`remainder_integer`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`select_one`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`set_driver_to_cpu`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`set_driver_to_gpu`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`sort_with_index`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`triton_builtin`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`unpack_flag`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`unpack_value`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`welford`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`welford_combine`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`welford_reduce`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`wrapper`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`x_grid_barrier`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)

### Imports

- **`.triton_compat`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`Any`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`Callable`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`collections.abc`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`driver`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`math`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`triton.runtime`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`typing`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)
- **`warnings`**: [triton_helpers.py_docs.md](./triton_helpers.py_docs.md)


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
