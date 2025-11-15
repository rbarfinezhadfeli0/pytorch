# Documentation: `docs/torch/_inductor/fx_passes/quantization.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/quantization.py_kw.md`
- **Size**: 13,068 bytes (12.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/fx_passes/quantization.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/quantization.py](../../../../torch/_inductor/fx_passes/quantization.py)
- **Documentation**: [`quantization.py_docs.md`](./quantization.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`PostOpAttr`**: [quantization.py_docs.md](./quantization.py_docs.md)

### Functions

- **`__init__`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_check_node_kwarg_arg_value`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_create_wgt_node`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_extra_check_fn`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_find_first_node_in_dequant_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_generate_dequant_bmm_node_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_generate_dequant_convolution_node_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_generate_dequant_linear_node_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_generate_linear_dynamic_fp16_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_generate_linear_t_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_generate_pattern_with_output_add`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_generate_qconv_weight_prepack_patterns`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_generate_qlinear_weight_prepack_patterns`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_get_linear_dq_node`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_get_linear_node`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_get_pattern_output_dtype`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_inner`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_int_mm_weight_prepack`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_input_output_same_scale_zp`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_concat_linear_int8_woq_optimization_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_concat_linear_woq_int4_fusion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_dequant_conv_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_dequant_linear_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_dequant_promotion_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_qconv_binary_optimization_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_qconv_lowering_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_qconv_post_op_fusion_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_qlinear_binary_optimization_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_qlinear_lowering_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_qlinear_post_op_fusion_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_quantized_conv_optimization_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_quantized_linear_optimization_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_quantized_maxpool2d_optimization_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_quantized_op_binary_optimization_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_is_valid_woq_optimization_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_may_generate_pattern_with_dtype_convert`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_may_generate_pattern_with_reshape`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_concat_linear_int8_woq_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_dequant_promotion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_dequant_promotion_pass`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_int8_woq_concat_linear_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_linear_dynamic_fp16_weight_prepack`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_linear_dynamic_fp16_weight_prepack_pass`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qconv_binary_fusion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qconv_post_op_fusion_pass`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qconv_unary_fusion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qconv_weight_prepack`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qconv_weight_prepack_pass`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qlinear_binary_fusion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qlinear_post_op_fusion_pass`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qlinear_unary_fusion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qlinear_weight_prepack`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_qlinear_weight_prepack_pass`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantization_binary_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantization_cat`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantization_lowerings`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantization_maxpool2d`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantization_reshape`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantization_unary_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantization_weight_pack_pass`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantized_cat_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantized_conv_binary_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantized_conv_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantized_linear_binary_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantized_linear_unary_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantized_maxpool2d_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_quantized_reshape_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_smooth_quant_int_mm_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_woq_lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_woq_lowerings`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_woq_mm_int8_pattern1`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_woq_mm_int8_pattern2`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_woq_mm_int8_pattern3`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_register_woq_mm_int8_pattern4`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_unary_fusion_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_validate_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_with_outer_reshape`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`clone_to_new_node`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`concat_linear_woq_int4`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`concat_wgt`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`dequant_promotion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`fn`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`generate_pattern_with_binary`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`generate_pattern_with_output_quant`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`generate_pattern_with_unary`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`get_dequantize_per_tensor_activation_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`get_pattern_no_bias`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`get_qconv2d_binary_pt2e_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`get_qconv_pt2e_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`get_qlinear_binary_pt2e_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`get_qlinear_pt2e_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`is_view_op`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`linear_dynamic_fp16_weight_prepack`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`maybe_replace_node`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qcat`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qconv`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qconv_binary`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qconv_weight_prepack`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qlinear`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qlinear_binary`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qlinear_post_op_fusion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qlinear_weight_prepack`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qmaxpool2d`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`qreshape`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`quant_lift_up`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`woq_int8`**: [quantization.py_docs.md](./quantization.py_docs.md)

### Imports

- **`..`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`..lowering`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`..pattern_matcher`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`..utils`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`.freezing_patterns`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`.mkldnn_fusion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`.post_grad`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`Any`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_can_be_inplace`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_get_remaining_users`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`_hardswish_fusion`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`config`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`copy`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`counters`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`functools`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`has_free_symbols`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`itertools`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`lowerings`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`map_arg`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`math`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`operator`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`pad_listlike`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`register_freezing_graph_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`register_lowering_pattern`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`torch`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`torch._dynamo.utils`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`torch.fx.node`**: [quantization.py_docs.md](./quantization.py_docs.md)
- **`typing`**: [quantization.py_docs.md](./quantization.py_docs.md)


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

- **File Documentation**: `quantization.py_kw.md_docs.md`
- **Keyword Index**: `quantization.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
