# Keyword Index: `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/transformers/cuda/sdp_utils.cpp](../../../../../../../aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- **Documentation**: [`sdp_utils.cpp_docs.md`](./sdp_utils.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/transformers/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`SMVersion`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)

### Functions

- **`can_use_cudnn_attention`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`can_use_flash_attention`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`can_use_mem_efficient_attention`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_all_tensors_on_device`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_cudnn_deterministic`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_cudnn_hardware_support`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_cudnn_layout`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_cudnn_tensor_shapes`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_dtypes_low_precision`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_flash_attention_hardware_support`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_flash_causal_non_square_seqlens`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_for_nested_inputs`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_for_seq_len_1_nested_tensor`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_head_dim_size_flash`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_head_dim_size_flash_nested`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_head_dim_size_mem_efficient`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_mem_efficient_hardware_support`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_prefer_cudnn_attention`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_runtime_disabled_cudnn`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`check_sm_version`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`constexpr`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`if`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`is_flash_attention_available`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`minimum_gemm_alignment`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`select_sdp_backend`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`use_tensor_cores`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/NestedTensorImpl.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/TensorSubclassLikeUtils.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/core/grad_mode.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/cuda/CUDAConfig.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/cuda/CUDAContext.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/cudnn/cudnn-wrapper.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/detail/CUDAHooksInterface.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/native/DispatchStub.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/native/transformers/cuda/sdp_utils.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/native/transformers/hip/aotriton_versions.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`ATen/native/transformers/sdp_utils_cpp.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`aotriton/flash.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`c10/core/ScalarType.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`c10/core/SymInt.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`c10/util/Array.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`c10/util/Exception.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`c10/util/env.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`c10/util/irange.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`c10/util/string_view.h`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)

### Namespaces

- **`bool`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)
- **`sdp`**: [sdp_utils.cpp_docs.md](./sdp_utils.cpp_docs.md)


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
