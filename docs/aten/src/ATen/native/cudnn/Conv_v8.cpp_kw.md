# Keyword Index: `aten/src/ATen/native/cudnn/Conv_v8.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/cudnn/Conv_v8.cpp](../../../../../../aten/src/ATen/native/cudnn/Conv_v8.cpp)
- **Documentation**: [`Conv_v8.cpp_docs.md`](./Conv_v8.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/cudnn`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`BenchmarkCache`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`CacheKey`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`CacheKeyFused`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`CacheKeyFusedWrapper`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`CacheKeyWrapper`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)

### Functions

- **`build_opgraph`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`build_opgraph_fused`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`filterEngineConfigs`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`generate_and_filter_plans`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`getAlignment`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`getConvDescriptor`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`getLRUCacheLimit`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`getTensorDescriptor`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`getTensorDescriptorWithTypeVirtual`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`get_available_workspace`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`get_configs_from_heuristics`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`get_configs_from_heuristics_fused`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`get_generator_sources`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`get_plans_from_find`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`get_plans_from_find_fused`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`if`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`plan_errata_exception`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`raw_cudnn_convolution_add_relu_out`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`raw_cudnn_convolution_backward_input_out`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`raw_cudnn_convolution_backward_weight_out`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`raw_cudnn_convolution_forward_out`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`run_conv_plan`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`run_conv_plan_fused`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`run_fused_conv`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`run_single_conv`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`try_configs`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`try_configs_fused`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`try_plans`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`try_plans_fused`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`update`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)

### Includes

- **`ATen/Functions.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/cuda/CUDAConfig.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/cuda/Exceptions.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/cudnn/Handle.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/cudnn/cudnn-wrapper.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/native/ConvUtils.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/native/cudnn/ConvShared.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/native/utils/ParamsHash.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`ATen/ops/empty.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`c10/cuda/CUDACachingAllocator.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`c10/cuda/CUDAException.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`c10/macros/Macros.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`c10/util/env.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`cudnn_frontend.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`cudnn_frontend_find_plan.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`cudnn_frontend_get_plan.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`dlfcn.h`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`list`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`unordered_map`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)

### Namespaces

- **`at`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`native`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)
- **`void`**: [Conv_v8.cpp_docs.md](./Conv_v8.cpp_docs.md)


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
