# Documentation: `docs/test/quantization/core/test_quantized_op.py_kw.md`

## File Metadata

- **Path**: `docs/test/quantization/core/test_quantized_op.py_kw.md`
- **Size**: 24,082 bytes (23.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Keyword Index: `test/quantization/core/test_quantized_op.py`

## File Information

- **Original File**: [test/quantization/core/test_quantized_op.py](../../../../test/quantization/core/test_quantized_op.py)
- **Documentation**: [`test_quantized_op.py_docs.md`](./test_quantized_op.py_docs.md)
- **Folder**: `test/quantization/core`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MultiheadAttentionModel`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`PointwisePostOp`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`QuantizableLSTMSplitGates`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestComparatorOps`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestDynamicQuantizedOps`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestPadding`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestQNNPackOps`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestQuantizedConv`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestQuantizedEmbeddingOps`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestQuantizedLinear`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestQuantizedOps`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`TestQuantizedWithMinMax`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`with`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)

### Functions

- **`__init__`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_dequantize_fp8e4m3`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_get_random_tensor_and_q_params`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_get_rnn_inputs`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_get_rnn_weights_and_bias`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_make_qconv_tensors`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_make_qconv_tensors_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_quantize_fp8e4m3`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_run_test`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_activation_function`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_binary_op_scalar_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_embedding_bag_unpack_fn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_embedding_bag_unpack_impl`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qconv_fp8_helper`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qconv_impl`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qconv_impl_cpu_tensor`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qconv_impl_cpu_tensor_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qconv_op_impl`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qconv_unpack_impl`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qlinear_fp8_helper`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qlinear_impl`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_test_qlinear_pt2e_helper`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`avoid_vpmaddubsw_overflow_linear`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`embedding_bag_rowwise_offsets_run`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`equal_ref`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`forward`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`from_float`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`func`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`get_reference_result`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`lengths_to_offsets`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`pool_output_shape`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`qlinear_ref`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_adaptive_avg_pool`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_adaptive_avg_pool2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_adaptive_avg_pool2d_nhwc`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_adaptive_avg_pool3d_ndhwc`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_add_scalar_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_advanced_indexing`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_avg_pool2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_avg_pool2d_nhwc`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_avg_pool3d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_avg_pool3d_nhwc`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_batch_norm`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_batch_norm_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_benchmark`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_cat`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_cat_nhwc`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_channel_shuffle`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_compare_tensor_scalar`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_compare_tensor_tensor`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_constant_padNd`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_conv_reorder_issue_onednn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_conv_transpose_reorder_issue_onednn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_custom_module_lstm`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_custom_module_multi_head_attention`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_dynamic_conv1d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_dynamic_conv2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_dynamic_conv3d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_dynamic_convtranspose1d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_dynamic_convtranspose2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_dynamic_convtranspose3d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_embedding`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_embedding_2d_indices`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_embedding_bag_2bit`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_embedding_bag_2d_indices`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_embedding_bag_4bit`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_embedding_bag_byte`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_empty_batch`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_equal`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_group_norm`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_hardswish`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_hardtanh`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_instance_norm`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_int8_add_onednn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_int8_batch_norm_onednn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_int8_mul_onednn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_interpolate`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_interpolate3d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_leaky_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_leaky_relu_observed_output`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_linear_bias_unpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_linear_dynamic_fp16_onednn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_linear_prepack_fp16_numerics`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_max_pool1d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_max_pool2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_max_pool2d_cudnn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_max_pool2d_nhwc`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_max_pool2d_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_max_pool3d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_max_pool3d_nhwc`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_mean`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_mul_scalar_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qadd_broadcast`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qadd_relu_cudnn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qadd_relu_cudnn_nhwc`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qadd_relu_different_qparams`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qadd_relu_same_qparams`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qcelu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qclamp`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d_cudnn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d_relu_cudnn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d_relu_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d_relu_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv1d_unpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_add`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_add_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_cudnn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_hardswish_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_hardswish_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_hardtanh_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_hardtanh_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_relu_cudnn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_relu_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_relu_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_sum_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_sum_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_sum_relu_float_output_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_sum_relu_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_sum_relu_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_swish_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_swish_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv2d_unpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv3d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv3d_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv3d_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv3d_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv3d_unpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv_transpose1d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv_transpose2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qconv_transpose3d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qelu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qgelu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qhardsigmoid`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlayer_norm`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_add_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_add_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_add_relu_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_add_relu_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_cudnn`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_dynamic_fp16`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_gelu_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_gelu_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_leaky_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_legacy`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_qnnpack_free_memory_and_unpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_relu_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_relu_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_sum_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_sum_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_sum_relu_fp8`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_sum_relu_pt2e`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_tanh`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_unpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlinear_with_input_q_dq_qweight_dq_output_fp32`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qlstmGRU`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qmatmul`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qmul_broadcast`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qmul_relu_different_qparams`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qmul_relu_same_qparams`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qnnpack_add`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qnnpack_add_broadcast`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qnnpack_maxpool2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qnnpack_mul`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qnnpack_relu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qnnpack_sigmoid`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qnnpack_sigmoid_sweep`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qnnpack_tanh`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qprelu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qrelu`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qrelu6`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qrnncell`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qsoftmax`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qsoftmax_qnnpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qtanh`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qthreshold`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_qtopk`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_quantize_tensor_with_min_max`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_quantized_equal`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_quantized_mean_qnnpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_reflection_pad1d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_reflection_pad2d`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_sigmoid`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_sigmoid_dequantize_rounding_error`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_sigmoid_non_observed`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_std`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_unpacked_qlinear_dynamic_fp16`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_unpacked_qlinear_dynamic_fp16_opcheck`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`test_wrapped_fbgemm_linear_fp16`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)

### Imports

- **`NamedTuple`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`OpOverloadPacket`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`Optional`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`PerChannelMinMaxObserver`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`ROCM_HOME`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`Version`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_VF`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`_pair`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`assume`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`copy`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`hypothesis`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`itertools`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`numpy`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`opcheck`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`operator`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`packaging.version`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`profile`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`random`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch._ops`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.ao.quantization`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.backends.xnnpack`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.jit`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.nn.functional`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.nn.modules.utils`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.profiler`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.testing._internal.common_quantization`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.testing._internal.common_quantized`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.testing._internal.hypothesis_utils`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.testing._internal.optests`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`torch.utils.cpp_extension`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`typing`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)
- **`unittest`**: [test_quantized_op.py_docs.md](./test_quantized_op.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/test/quantization/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/core`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/quantization/core/test_quantized_op.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/core`):

- [`test_workflow_module.py_kw.md_docs.md`](./test_workflow_module.py_kw.md_docs.md)
- [`test_quantized_tensor.py_kw.md_docs.md`](./test_quantized_tensor.py_kw.md_docs.md)
- [`test_backend_config.py_docs.md_docs.md`](./test_backend_config.py_docs.md_docs.md)
- [`test_workflow_module.py_docs.md_docs.md`](./test_workflow_module.py_docs.md_docs.md)
- [`test_top_level_apis.py_docs.md_docs.md`](./test_top_level_apis.py_docs.md_docs.md)
- [`test_quantized_module.py_docs.md_docs.md`](./test_quantized_module.py_docs.md_docs.md)
- [`test_quantized_functional.py_kw.md_docs.md`](./test_quantized_functional.py_kw.md_docs.md)
- [`test_utils.py_kw.md_docs.md`](./test_utils.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_quantized_op.py_kw.md_docs.md`
- **Keyword Index**: `test_quantized_op.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
