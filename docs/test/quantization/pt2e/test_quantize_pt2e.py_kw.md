# Keyword Index: `test/quantization/pt2e/test_quantize_pt2e.py`

## File Information

- **Original File**: [test/quantization/pt2e/test_quantize_pt2e.py](../../../../test/quantization/pt2e/test_quantize_pt2e.py)
- **Documentation**: [`test_quantize_pt2e.py_docs.md`](./test_quantize_pt2e.py_docs.md)
- **Folder**: `test/quantization/pt2e`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BackendAQuantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`BadQuantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`DtypeActQuantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`Int4Observer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`M`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`Model`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`TestQuantizePT2E`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`TestQuantizePT2EAffineQuantization`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`TestQuantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`TestQuantizer1`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`TestQuantizer2`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)

### Functions

- **`__init__`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`_assert_ops_are_correct`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`_fake_recompile`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`_get_bn_train_eval_ops`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`_get_node`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`_test_fixed_qparams_qspec`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`_test_move_exported_model_dropout`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`_test_transitive_sharing_with_cat_helper`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`annotate`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`calculate_qparams`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`check_nn_module`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`convert`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`dequantize_per_tensor_int4`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`derive_qparams_fn`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`dynamic_quantize_pt2e`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`example_inputs`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`forward`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`prepare_obs_or_fq_callback`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`quantize_per_tensor_int4`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_allow_exported_model_train_eval`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_allow_exported_model_train_eval_idempotent`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_allow_implicit_sharing`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_channel_group_quantization`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_composable_quantizer_linear_conv`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_composable_quantizer_throw`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_composable_quantizer_transform_for_annotation`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_constant_prop_preserve_metadata`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_conv_padding_bn_relu`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_conv_transpose_bn_relu`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_derived_qspec`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_derived_qspec_per_channel`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_disallow_eval_train`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_dont_fold_other_constant`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_dynamic_affine_act_per_channel_weights`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_dynamic_per_tok_act_per_group_weights`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_embedding_conv_linear_quantization`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_embedding_quantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_fixed_qparams_qspec_observer_dedup`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_fixed_qparams_qspec_ptq`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_fixed_qparams_qspec_qat`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_fold_all_ops_before_quantize`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_fold_quantize`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_fold_quantize_per_channel`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_groupwise_per_channel_quant`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_input_edge_sanity_check`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_max_pool2d_quantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_model_is_exported`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_move_exported_model_bn`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_move_exported_model_dropout`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_move_exported_model_dropout_inplace`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_multi_users_without_output_observer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_observer_callback`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_prepare_obs_or_fq_callback`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_preserve_nn_module_stack`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_quantization_dtype`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_reentrant`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_save_load`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_shared_qspec`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_shared_qspec_transitivity`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_shared_qspec_transitivity_case_2`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_simple_quantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_speed`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_transform_for_annotation`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`test_wo_annotate_conv_output_quantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`transform_for_annotation`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`validate`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`verify_quant_dequant_iotypes`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)

### Imports

- **`MappingType`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`Node`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`ObserverBase`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`Tensor`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`custom_op`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`export`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`observer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`operator`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`time`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.observer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.pt2e._affine_quantization`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.quantize_pt2e`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.quantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.quantizer.composable_quantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.quantizer.embedding_quantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.quantizer.xnnpack_quantizer`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.ao.quantization.quantizer.xnnpack_quantizer_utils`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.export`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.fx`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.library`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.testing._internal.common_quantization`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_quantize_pt2e.py_docs.md](./test_quantize_pt2e.py_docs.md)


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
