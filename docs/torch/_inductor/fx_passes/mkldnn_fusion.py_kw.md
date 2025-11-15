# Keyword Index: `torch/_inductor/fx_passes/mkldnn_fusion.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/mkldnn_fusion.py](../../../../torch/_inductor/fx_passes/mkldnn_fusion.py)
- **Documentation**: [`mkldnn_fusion.py_docs.md`](./mkldnn_fusion.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CpuMkldnnDeviceOp`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`MkldnnDeviceOpBase`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`Model`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`UnaryAttr`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`XpuMkldnnDeviceOp`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`based`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)

### Functions

- **`__init__`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_binary_fusion_v1`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_binary_fusion_v2`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_can_be_inplace`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_check_input_sizes`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_combined_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_conv_call`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_conv_transpose_call`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_eliminate_duplicate_packed_nodes`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_gelu_fusion_1`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_gelu_fusion_2`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_get_compute_node`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_get_mkldnn_device_op`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_get_remaining_users`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_hardsigmoid_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_hardswish_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_hardtanh_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_ancestor_node`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_packable_convolution`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_packable_linear`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_packable_mkldnn_rnn_layer`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_single_computation_op`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_valid_binary`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_valid_computation_binary`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_valid_computation_binary_inplace`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_valid_computation_unary_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_is_valid_grouped_gemm_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_leaky_relu_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_linear_call`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_mkldnn_fusion_init`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_mkldnn_weight_pack_init`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_other_input_not_inplaceable`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_recover_linear`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_binary_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_binary_unary_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_binary_unary_fusion_lowering`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_binary_unary_maybe_inplace_fusion_lowering`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_hardtanh_fusion_lowering`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_inplace_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_leaky_relu_fusion_lowering`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_unary_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_unary_fusion_lowering`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_register_weight_pack_pass`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_silu_fusion`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_to_bf16`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_to_float`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_to_fp16`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_unary_fusion_pattern`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`_unary_fusion_patterns`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`convolution`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`fn`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`forward`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`get_item`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`get_linear_transpose_weight`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`get_meta_value`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`get_val`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`grouped_gemm_pass`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`is_const_or_cat_by_const`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`is_linear_add_bias`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`linear`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`linear_bias_pattern`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`mkldnn_rnn_layer`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`pack_conv_weight`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`pack_linear`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`pack_linear_weight`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`reshape_linear_reshape_pattern`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)

### Imports

- **`..`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`..lowering`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`..mkldnn_lowerings`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`..pattern_matcher`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`..utils`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`..virtualized`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`.freezing_patterns`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`.post_grad`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`.quantization`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`Any`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`Callable`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`OrderedSet`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`collections.abc`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`counters`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`functools`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`grouped_gemm_lowering`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`has_free_symbols`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`ir`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`lowerings`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`operator`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`ops`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`reduce`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`register_freezing_graph_pattern`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`register_lowering_pattern`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`torch`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`torch._dynamo.utils`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`torch.utils._ordered_set`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)
- **`typing`**: [mkldnn_fusion.py_docs.md](./mkldnn_fusion.py_docs.md)


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
