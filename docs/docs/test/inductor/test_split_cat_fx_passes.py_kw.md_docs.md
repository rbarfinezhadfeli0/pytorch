# Documentation: `docs/test/inductor/test_split_cat_fx_passes.py_kw.md`

## File Metadata

- **Path**: `docs/test/inductor/test_split_cat_fx_passes.py_kw.md`
- **Size**: 13,841 bytes (13.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Keyword Index: `test/inductor/test_split_cat_fx_passes.py`

## File Information

- **Original File**: [test/inductor/test_split_cat_fx_passes.py](../../../test/inductor/test_split_cat_fx_passes.py)
- **Documentation**: [`test_split_cat_fx_passes.py_docs.md`](./test_split_cat_fx_passes.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestSplitCatFxPasses`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)

### Functions

- **`arg_only`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`arg_only_cm`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`arg_only_dim0`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`caoncat_only`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`cm_with_list`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`diff_dims`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`dim_mismatch`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`duplicate_getitems`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`duplicate_getitems_neg_index`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`fn`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`graph_should_be_topological_sorted`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`input_shuffling`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`input_shuffling_dim_mismatch`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`input_shuffling_dim_mismatch_stack`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`input_shuffling_direct_output`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`input_shuffling_multiple_output`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`input_shuffling_multiple_output_same_ranges`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`input_shuffling_stack`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`kwarg1`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`kwarg1_cm`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`kwarg2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`kwarg2_cm`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`kwarg3`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`list_replace`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`move_reshape_out_of_split_stack`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`multi_split`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`multi_split_2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`multi_split_2_neg_dim`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`multi_split_cat`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`multi_split_cm`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`multi_split_kwarg1`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`multi_split_kwarg2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`multi_split_with_sizes`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`mutate_cat_node_with_some_getitmes`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`next_split_getitem_partial_used`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`normalize_reshape_with_dynamic_shape`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`other_users`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`other_users_2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`patch`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`remove_cat_node_with_all_getitmes`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`simple_split_cat`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`simple_split_cat_argspec1`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`simple_split_cat_argspec2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`simple_split_cat_argspec3`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`simple_split_cat_argspec4`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`simple_split_stack`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`simple_split_stack_argspec1`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`simple_split_stack_argspec2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`some_users_not_splits`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_addn_args`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_addn_args_dim2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_dim_mismatch`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_dim_mismatch2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_dim_mismatch3`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_mutation`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_split`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_split_kwarg`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_cat_to_slices`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_getitem_gap`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_getitem_out_of_order`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_partial_getitem_cat`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_size_not_1`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_squeeze_multi_squeeze_users`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_squeeze_stack`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_squeeze_stack_callmethod`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_squeeze_stack_callmethod_none_dim`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_squeeze_stack_kwarg1`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_squeeze_stack_kwarg1_callmethod`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_stack_addn_args`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_stack_dim_mismatch`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_stack_dim_mismatch2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_stack_dim_mismatch3`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_stack_to_cats_different_dim`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_stack_to_cats_same_dim`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`split_with_cat`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_cat_normalization`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_config_flag_is_respected`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_consecutive_split_merge`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_numpy_compat_normalization`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_split_cat_merge`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_split_cat_merge_mutation`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_split_cat_new_patterns`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_split_normalization`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_split_squeeze`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_stack_normalization_axis_kwarg`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`test_unbind_stack`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_cat`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_cat_addn_args`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_cat_addn_args_dim2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_cat_dim_mismatch`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_cat_multi_users`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_cat_multi_users_diff_dims`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_cat_to_view`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_stack`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_stack_addn_args`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_stack_argspec1`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_stack_argspec2`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_stack_dim_mismatch`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unbind_stack_to_slices`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unequal_multi_split`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unequal_multi_split_neg_index`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unequal_split`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unequal_split_cm`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`unequal_split_multiple_output`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)

### Imports

- **`GPU_TYPE`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`IS_LINUX`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`counters`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`numpy_compat_normalization`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`requires_gpu`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`run_tests`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`torch`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`torch._dynamo.utils`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`torch._inductor.fx_passes.misc_patterns`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`torch._inductor.test_case`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)
- **`torch.testing._internal.triton_utils`**: [test_split_cat_fx_passes.py_docs.md](./test_split_cat_fx_passes.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/inductor/test_split_cat_fx_passes.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_split_cat_fx_passes.py_kw.md_docs.md`
- **Keyword Index**: `test_split_cat_fx_passes.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
