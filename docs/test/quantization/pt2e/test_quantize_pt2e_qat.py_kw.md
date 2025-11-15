# Keyword Index: `test/quantization/pt2e/test_quantize_pt2e_qat.py`

## File Information

- **Original File**: [test/quantization/pt2e/test_quantize_pt2e_qat.py](../../../../test/quantization/pt2e/test_quantize_pt2e_qat.py)
- **Documentation**: [`test_quantize_pt2e_qat.py_docs.md`](./test_quantize_pt2e_qat.py_docs.md)
- **Folder**: `test/quantization/pt2e`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ConvBnDerivedBiasQuantizer`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`ConvBnInt32WeightQuantizer`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`M`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`M2`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`PT2EQATTestCase`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`QATPTQTestModule`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`TestQuantizeMixQATAndPTQ`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`TestQuantizePT2EQATModels`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`TestQuantizePT2EQAT_ConvBn1d`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`TestQuantizePT2EQAT_ConvBn2d`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`TestQuantizePT2EQAT_ConvBn_Base`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`TwoLinear`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_BaseConvBnModel`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`if`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`which`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)

### Functions

- **`__init__`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_convert_qat_linears`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_derive_bias_qparams_from_act_and_weight_qparams`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_do_test_qat_conv_transpose_bn`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_get_conv_bn_getitem_nodes`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_get_conv_bn_model`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_has_add_`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_is_conv_node`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_prepare_qat_linears`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_verify_symmetric_xnnpack_qat_graph`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_verify_symmetric_xnnpack_qat_graph_helper`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_verify_symmetric_xnnpack_qat_numerics`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`_verify_symmetric_xnnpack_qat_numerics_helper`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`annotate`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`forward`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`get_conv_weight_and_bias`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`get_source_fn`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`setUp`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_fold_bn_erases_add_node`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_fold_bn_erases_bn_node`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_mixing_qat_ptq`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_bias_derived_qspec`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_fusion`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_fusion_cuda`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_fusion_literal_args`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_fusion_no_conv_bias`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_per_channel_weight_bias`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_relu_fusion`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_relu_fusion_cuda`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_bn_relu_fusion_no_conv_bias`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_no_bias`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_transpose_bn`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_conv_transpose_bn_relu`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_inplace_add_relu`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_mobilenet_v2`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_per_channel_weight_custom_dtype`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_preserve_source_fn_stack`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_resnet18`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`test_qat_update_shared_qspec`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`validate`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)

### Imports

- **`Any`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`TEST_CUDA`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`copy`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`export`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`get_qnnpack_backend_config`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`operator`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`override_quantized_engine`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`prepare_qat_fx`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`raise_on_run_directly`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.ao.quantization`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.ao.quantization.backend_config`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.ao.quantization.quantize_fx`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.ao.quantization.quantize_pt2e`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.ao.quantization.quantizer`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.ao.quantization.quantizer.xnnpack_quantizer`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.export`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.testing._internal.common_quantization`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.testing._internal.common_quantized`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`torchvision`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`typing`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)
- **`unittest`**: [test_quantize_pt2e_qat.py_docs.md](./test_quantize_pt2e_qat.py_docs.md)


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
