# Keyword Index: `test/fx/test_fx_const_fold.py`

## File Information

- **Original File**: [test/fx/test_fx_const_fold.py](../../../test/fx/test_fx_const_fold.py)
- **Documentation**: [`test_fx_const_fold.py_docs.md`](./test_fx_const_fold.py_docs.md)
- **Folder**: `test/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ConstFoldTestModule`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`SubModule`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`TestConstFold`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`TestModule`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`TracedThroughModule`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)

### Functions

- **`__init__`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`_get_attr`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`_test_const_fold_tensor_meta`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`_verify_const_fold_mod`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`forward`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`skip_folding_quant_dequant`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_check_inline_non_const`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_check_inline_non_const_mult_return`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_check_skip_folding_quant_dequant_pattern`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_basic_one_attr_name_collision`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_basic_one_attr_no_name_collision`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_basic_placeholder_reordered`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_basic_two_attr`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_basic_two_attr_three_input`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_has_inlined_call_module_node`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_module_attr`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_multi_const_folded_attrs`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_noop`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_partial_graph`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_submod_hierarchy`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_tensor_meta`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_const_fold_unused_placeholder`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_dict_output`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_do_not_fold_impure_subgraph`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_fold_module`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_fold_pure_subgraph`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_retain_node_meta`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_three_outputs`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`test_two_outputs`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)

### Imports

- **`_extract_tensor_metadata`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`const_fold`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`operator`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`raise_on_run_directly`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`torch`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`torch.fx`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`torch.fx.experimental`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`torch.fx.passes.shape_prop`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fx_const_fold.py_docs.md](./test_fx_const_fold.py_docs.md)


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
