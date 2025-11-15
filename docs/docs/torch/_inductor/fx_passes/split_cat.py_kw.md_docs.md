# Documentation: `docs/torch/_inductor/fx_passes/split_cat.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/split_cat.py_kw.md`
- **Size**: 7,882 bytes (7.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/fx_passes/split_cat.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/split_cat.py](../../../../torch/_inductor/fx_passes/split_cat.py)
- **Documentation**: [`split_cat.py_docs.md`](./split_cat.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GetItem`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`SplitCatSimplifier`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`TorchSplit`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`UnbindCatRemover`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`to`**: [split_cat.py_docs.md](./split_cat.py_docs.md)

### Functions

- **`__init__`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`_get_dim`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`_get_split_args_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`_match`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`calculate_fused_tensor_size`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`construct_cat_args`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`construct_pattern_matcher_pass`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`convert_reshape_cat_arg_to_stack`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`divide_into_consecutive_sublists`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`erase_old_nodes`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`fill_gaps`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`find_anchor_nodes`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`find_next_users`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`get_merged_user_inputs`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`get_non_cat_node_input`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`get_simplified_split_ranges`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`get_transform_params`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`get_user_input_list`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`get_view_shape_list`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`has_non_overlapping_ranges`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`has_same_parent_node`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`is_empty_tensor`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`is_sorted_and_consecutive`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`match_einsum_strings`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`merge_consecutive_inputs`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`merge_getitem_cat`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`merge_select_cat_aten`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`merge_split_cat_aten`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`merge_split_squeeze`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`merge_splits`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`merge_unbind_stack`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`merge_unbind_stack_aten`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`move_reshape_out_of_split_stack`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`move_view_after_cat`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`mutate_cat_node`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_cat_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_cat_default_aten`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_clamp_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_detach_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_reshape_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_split_base`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_split_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_split_default_aten`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_split_with_size_default_aten`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_squeeze_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_stack_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`normalize_unbind_default`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`remove_split_unbind_children`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`remove_split_with_size_one`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`remove_unbind`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`remove_zeros`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`repl`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`replace_cat`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`replace_einsum_to_pointwise`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`replace_split`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`reshape_cat_node`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`reshape_cat_node_to_stack`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`should_replace_einsum`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`simplify`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`simplify_split_cat`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`split_cat_to_slices`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`split_stack_to_cats`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`unbind_cat_to_view`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`unbind_stack_to_slices`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`update_args_from_split_getitem`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`update_args_from_unbind_getitem`**: [split_cat.py_docs.md](./split_cat.py_docs.md)

### Imports

- **`..pattern_matcher`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`.group_batch_fusion`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`Any`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`Callable`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`OrderedSet`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`collections`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`collections.abc`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`counters`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`defaultdict`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`free_symbols`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`is_node_meta_valid`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`itertools`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`logging`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`operator`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`os`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`torch`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`torch._dynamo.utils`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`torch.utils._ordered_set`**: [split_cat.py_docs.md](./split_cat.py_docs.md)
- **`typing`**: [split_cat.py_docs.md](./split_cat.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_inductor/fx_passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_inductor/fx_passes`):

- [`dedupe_symint_uses.py_kw.md_docs.md`](./dedupe_symint_uses.py_kw.md_docs.md)
- [`overlap_preserving_bucketer.py_kw.md_docs.md`](./overlap_preserving_bucketer.py_kw.md_docs.md)
- [`pre_grad.py_docs.md_docs.md`](./pre_grad.py_docs.md_docs.md)
- [`b2b_gemm.py_docs.md_docs.md`](./b2b_gemm.py_docs.md_docs.md)
- [`freezing_patterns.py_kw.md_docs.md`](./freezing_patterns.py_kw.md_docs.md)
- [`fsdp.py_docs.md_docs.md`](./fsdp.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`replace_random.py_kw.md_docs.md`](./replace_random.py_kw.md_docs.md)
- [`joint_graph.py_kw.md_docs.md`](./joint_graph.py_kw.md_docs.md)
- [`numeric_utils.py_docs.md_docs.md`](./numeric_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `split_cat.py_kw.md_docs.md`
- **Keyword Index**: `split_cat.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
