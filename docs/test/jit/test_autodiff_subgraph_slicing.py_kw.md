# Keyword Index: `test/jit/test_autodiff_subgraph_slicing.py`

## File Information

- **Original File**: [test/jit/test_autodiff_subgraph_slicing.py](../../../test/jit/test_autodiff_subgraph_slicing.py)
- **Documentation**: [`test_autodiff_subgraph_slicing.py_docs.md`](./test_autodiff_subgraph_slicing.py_docs.md)
- **Folder**: `test/jit`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`M`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`TestAutodiffSubgraphSlicing`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)

### Functions

- **`__init__`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`_perform_ad_subgraph_slicing`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`assertGraphSize`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`bar`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`fn`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`foo`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`forward`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`func`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`method1`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`t`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_aliased_outputs`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_bias_as_arg`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_bias_as_module_attr`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_chunk_constant_script_ad`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_constructed_bias`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_diff_graph_inline_threshold`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_differentiable_graph_ops_requires_grad`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_does_not_create_cycles`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_does_not_merge_unrelated`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_has_profiled_info_aliasing_outputs`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_merge_respects_aliasing`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_merges_dense`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_merges_down`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_merges_up`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_merges_without_cycles`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_prune_grad`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_requires_grad_for_tensor_list`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_respects_lexical_scoping`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_simple_merge`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`test_simple_no_merge`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)

### Imports

- **`FileCheck`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`List`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`check_against_reference`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`os`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`raise_on_run_directly`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`sys`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`torch`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`torch.testing`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`torch.testing._internal.common_jit`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`torch.testing._internal.jit_utils`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`typing`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)
- **`unittest`**: [test_autodiff_subgraph_slicing.py_docs.md](./test_autodiff_subgraph_slicing.py_docs.md)


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
