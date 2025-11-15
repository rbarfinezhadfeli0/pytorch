# Keyword Index: `aten/src/ATen/native/transformers/attention.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/transformers/attention.cpp](../../../../../../aten/src/ATen/native/transformers/attention.cpp)
- **Documentation**: [`attention.cpp_docs.md`](./attention.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/transformers`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_fused_sdp_choice_cpp`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`_fused_sdp_choice_meta`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`_safe_softmax`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`aligned_tensor`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`bmm_nn`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`bmm_nt`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`debug_assert_shape`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`gemm_nt`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`if`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`masked_softmax`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`pad_bias`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`pad_last_dim`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`post_process_flash_output`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`preprocess_mask`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`qkv_projection`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`scaled_dot_product_attention`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`should_compute_logsumexp`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`transform0213_gemm_nt_bias`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`transform_0213`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`triton_multi_head_attention`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`validate_sdpa_input`**: [attention.cpp_docs.md](./attention.cpp_docs.md)

### Includes

- **`ATen/AccumulateType.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/Dispatch.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/Functions.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/NestedTensorImpl.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/OpMathType.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/TensorIndexing.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/TensorOperators.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/TensorSubclassLikeUtils.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/core/TensorBody.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/mps/MPSDevice.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/native/DispatchStub.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/native/nested/NestedTensorTransformerFunctions.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/native/transformers/attention.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/native/transformers/sdp_utils_cpp.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_fused_sdp_choice_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_fused_sdp_choice_ops.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_masked_softmax.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_native_multi_head_attention_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_nested_from_padded.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_nested_tensor_softmax_with_shape.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_safe_softmax.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_safe_softmax_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_attention_math.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_attention_math_for_mps.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_attention_math_for_mps_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_attention_math_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_cudnn_attention.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_efficient_attention.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention_backward_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention_for_cpu.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention_for_cpu_backward.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention_for_cpu_backward_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention_for_cpu_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_fused_attention_overrideable.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_fused_attention_overrideable_backward.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_fused_attention_overrideable_backward_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_scaled_dot_product_fused_attention_overrideable_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_softmax.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_transform_bias_rescale_qkv.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_transform_bias_rescale_qkv_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_triton_multi_head_attention_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/_triton_scaled_dot_attention.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/all.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/bmm.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/cat.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/chunk_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/dropout.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/linear_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/matmul.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/matmul_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/ones.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/pad.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/scaled_dot_product_attention_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/softmax.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/split_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/split_with_sizes_native.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/where.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`ATen/ops/zeros_like.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`c10/core/DeviceType.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`c10/core/DispatchKey.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`c10/core/DispatchKeySet.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`c10/core/SymInt.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`c10/core/SymIntArrayRef.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`c10/util/Logging.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`c10/util/typeid.h`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`limits`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`utility`**: [attention.cpp_docs.md](./attention.cpp_docs.md)

### Namespaces

- **`Tensor`**: [attention.cpp_docs.md](./attention.cpp_docs.md)
- **`at`**: [attention.cpp_docs.md](./attention.cpp_docs.md)


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
