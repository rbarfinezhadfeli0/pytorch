# Keyword Index: `torch/_inductor/autoheuristic/autoheuristic_utils.py`

## File Information

- **Original File**: [torch/_inductor/autoheuristic/autoheuristic_utils.py](../../../../torch/_inductor/autoheuristic/autoheuristic_utils.py)
- **Documentation**: [`autoheuristic_utils.py_docs.md`](./autoheuristic_utils.py_docs.md)
- **Folder**: `torch/_inductor/autoheuristic`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AHContext`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`AHFeature`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`AHMetadata`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`AHOperation`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`is`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)

### Functions

- **`__init__`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`add_feature`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`apply_operation`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`apply_operations`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`between_op`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`between_ops`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`bfloat_perf_hit`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`check_minsize`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`context_add_strides`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`context_add_using_tf32`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_arith_intensity`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_dims_multiple_ops`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_dims_need_padding_ops`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_feature_names_csv`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_feature_values_csv`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_is_contig_ops`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_metadata_str_from_log`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_mixedmm_precondition`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_mult_dims_ops`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_numerical_and_categorical_features`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`get_value`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`is_multiple`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`mat1_innermost_needs_padding_fn`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`mat1_is_contig_fn`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`mat2_innermost_needs_padding_fn`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`mat2_is_contig_fn`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`mixed_mm_operations`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`mm_operations`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`num_dims_needs_padding_fn`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`pad_mm_operations`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`pad_mm_precondition`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`pow2_op`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`to_dict`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)

### Imports

- **`Any`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`Callable`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`collections.abc`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`functools`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`torch`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)
- **`typing`**: [autoheuristic_utils.py_docs.md](./autoheuristic_utils.py_docs.md)


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
