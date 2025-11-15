# Documentation: `docs/torch/_inductor/fx_passes/pre_grad.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/pre_grad.py_kw.md`
- **Size**: 7,856 bytes (7.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/fx_passes/pre_grad.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/pre_grad.py](../../../../torch/_inductor/fx_passes/pre_grad.py)
- **Documentation**: [`pre_grad.py_docs.md`](./pre_grad.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ConvBNFusion`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`IdentityRemover`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`NormalizedLinearNode`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`NormalizedMatmulNode`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)

### Functions

- **`__init__`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`_get_pass_name_func`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`_run_pre_dispatch_passes`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`_used_by_same_conv_module`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`add_bn_node`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`call_module`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`cat_args`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`check_permute`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`disable_fusion`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fetch_attr`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fuse_chunk_reshape_concat_pass`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fuse_chunk_reshape_unsqueeze_concat_pass`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fuse_conv_bn`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fuse_fx`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fuse_parallel_linear_pass`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fuse_split_getitem_squeeze_cat`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`get_bias`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`get_input`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`get_other`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`get_weight`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`is_fusion_enabled`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`is_pointwise_unary`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`is_same_dict`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`is_view`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`lazy_init`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`linear_permute_fusion`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`linear_transpose`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`merge_concats_pass`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`normalize_node_kwargs_pass`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`one_user`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`permute_linear_fusion`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`permute_matmul_fusion`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`pre_grad_passes`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`relu_nan_to_num`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`remove_identity`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`remove_noop_pass`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`remove_split_ops`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`remove_split_ops_pass`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`save_inductor_dict`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`shape_prop`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`sink_cat_after_pointwise`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`stack_to_unsqueeze_pass`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`transpose_linear`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`transpose_matmul`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`use_matmul_fuse_lce_replace_first_LCE`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`use_matmul_lce_replace_normal_LCE`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`use_triton_dot_compress`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`use_triton_lce_replace_normal_LCE`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`use_triton_lce_replace_normal_LCE_helper`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`use_triton_lce_replace_simple_LCE`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`use_triton_lce_replace_simple_LCE_helper`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)

### Imports

- **`.`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`..`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`..fx_utils`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`..pattern_matcher`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`..utils`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`.group_batch_fusion`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`.misc_patterns`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`.numeric_utils`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`.quantization`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`.split_cat`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`PRE_GRAD_PATTERNS`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`Sequence`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`ShapeProp`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`collections.abc`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`config`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`copy`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`counters`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`efficient_conv_bn_eval`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fb`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`functional`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`functools`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`fuse_conv_bn_eval`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`group_batch_fusion_passes`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`is_cpu_device`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`itertools`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`logging`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`matches_module_function_pattern`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`numeric_check_if_enabled`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`numpy_compat_normalization`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`quant_lift_up`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`torch`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`torch._dynamo.utils`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`torch._logging`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`torch.fx.experimental.optimization`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`torch.fx.passes.graph_transform_observer`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`torch.fx.passes.shape_prop`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`torch.nn`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`torch.nn.utils.fusion`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`trace_structured`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)
- **`types`**: [pre_grad.py_docs.md](./pre_grad.py_docs.md)


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

- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `pre_grad.py_kw.md_docs.md`
- **Keyword Index**: `pre_grad.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
