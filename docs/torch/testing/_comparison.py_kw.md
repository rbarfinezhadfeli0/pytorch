# Keyword Index: `torch/testing/_comparison.py`

## File Information

- **Original File**: [torch/testing/_comparison.py](../../../torch/testing/_comparison.py)
- **Documentation**: [`_comparison.py_docs.md`](./_comparison.py_docs.md)
- **Folder**: `torch/testing`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BooleanPair`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`ErrorMeta`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`NonePair`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`NumberPair`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`ObjectPair`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`Pair`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`TensorLikePair`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`UnsupportedInputs`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`and`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`needs`**: [_comparison.py_docs.md](./_comparison.py_docs.md)

### Functions

- **`__init__`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`__repr__`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_check_inputs_isinstance`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_check_supported`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_compare_attributes`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_compare_quantized_values`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_compare_regular_values_close`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_compare_regular_values_equal`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_compare_sparse_compressed_values`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_compare_sparse_coo_values`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_compare_values`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_equalize_attributes`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_fail`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_inputs_not_supported`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_make_bitwise_mismatch_msg`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_make_mismatch_msg`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_process_inputs`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_supported_types`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_to_bool`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_to_number`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`_to_tensor`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`assert_allclose`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`assert_close`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`bitwise_comp`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`compare`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`default_tolerances`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`extra_repr`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`get_tolerances`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`make_diff_msg`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`make_scalar_mismatch_msg`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`make_tensor_mismatch_msg`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`not_close_error_metas`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`originate_pairs`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`raise_mismatch_error`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`to_error`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`unravel_flat_index`**: [_comparison.py_docs.md](./_comparison.py_docs.md)

### Imports

- **`Any`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`Callable`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`OrderedDict`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`abc`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`cmath`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`collections`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`collections.abc`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`contextlib`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`deprecated`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`functools`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`math`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`numpy`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`torch`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`typing`**: [_comparison.py_docs.md](./_comparison.py_docs.md)
- **`typing_extensions`**: [_comparison.py_docs.md](./_comparison.py_docs.md)


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
