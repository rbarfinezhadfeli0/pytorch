# Keyword Index: `aten/src/ATen/native/transformers/cuda/attention.cu`

## File Information

- **Original File**: [aten/src/ATen/native/transformers/cuda/attention.cu](../../../../../../../aten/src/ATen/native/transformers/cuda/attention.cu)
- **Documentation**: [`attention.cu_docs.md`](./attention.cu_docs.md)
- **Folder**: `aten/src/ATen/native/transformers/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`a`**: [attention.cu_docs.md](./attention.cu_docs.md)

### Functions

- **`DISPATCH_TYPES`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`_fused_sdp_choice_cuda`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`collapse_dims_1_and_2`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`constexpr`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`for`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`if`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`rand_uniform_kernel`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`transform_bias_rescale_qkv_add_padding_kernel`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`transform_bias_rescale_qkv_kernel`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`triton_scaled_dot_attention`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`unpack_cudnn`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`unpack_cudnn_wrapper`**: [attention.cu_docs.md](./attention.cu_docs.md)

### Includes

- **`ATen/AccumulateType.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/Dispatch.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/Functions.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/NativeFunctions.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/NestedTensorImpl.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/TensorAccessor.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/TensorOperators.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/core/Tensor.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/cuda/CUDAContext.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/cuda/CUDAGraphsUtils.cuh`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/cuda/detail/IndexUtils.cuh`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/cuda/detail/KernelUtils.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/DispatchStub.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/NonSymbolicBC.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/cuda/Loops.cuh`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/cuda/MemoryAccess.cuh`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/cuda/PersistentSoftmax.cuh`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/cuda/block_reduce.cuh`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/cudnn/MHA.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/cudnn/hip/MHA.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/nested/NestedTensorTransformerFunctions.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/nested/NestedTensorTransformerUtils.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/nested/NestedTensorUtils.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/attention.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/cuda/flash_attn/flash_api.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassF.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/cuda/mem_eff_attention/pytorch_utils.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/cuda/sdp_utils.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/hip/aotriton_adapter.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/hip/flash_attn/ck/me_ck_api.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/native/transformers/sdp_utils_cpp.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_cudnn_attention_forward.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_cudnn_attention_forward_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_efficient_attention_forward.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_efficient_attention_forward_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_fill_mem_eff_dropout_mask_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_flash_attention_forward.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_flash_attention_forward_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_fused_sdp_choice_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_masked_softmax.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_native_multi_head_attention_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_scaled_dot_product_efficient_attention.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_scaled_dot_product_efficient_attention_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_scaled_dot_product_flash_attention_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_softmax.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_transform_bias_rescale_qkv.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_triton_multi_head_attention_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/_triton_scaled_dot_attention.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/empty.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/empty_like.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/empty_strided.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/linear.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/narrow_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/scalar_tensor.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/scaled_dot_product_attention.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/scaled_dot_product_attention_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/split_native.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`ATen/ops/zeros.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`aotriton/flash.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`aotriton/runtime.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`c10/cuda/CUDAMathCompat.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`c10/util/Logging.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`c10/util/bit_cast.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`namespace_config.h`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`optional`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`type_traits`**: [attention.cu_docs.md](./attention.cu_docs.md)

### Namespaces

- **`at`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`cuda`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`native`**: [attention.cu_docs.md](./attention.cu_docs.md)
- **`pytorch_flash`**: [attention.cu_docs.md](./attention.cu_docs.md)


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
