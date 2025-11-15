# Keyword Index: `aten/src/ATen/native/miopen/Conv_miopen.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/miopen/Conv_miopen.cpp](../../../../../../aten/src/ATen/native/miopen/Conv_miopen.cpp)
- **Documentation**: [`Conv_miopen.cpp_docs.md`](./Conv_miopen.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/miopen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`BenchmarkCache`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ConvolutionArgs`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ConvolutionParams`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`Workspace`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`algorithm_search`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`for`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`is`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)

### Functions

- **`chooseAlgorithm`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`chooseSolution`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`find`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`findAlgorithm`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`for`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`getBestAlgorithm`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`getSolution`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`getWorkspaceSize`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`if`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`insert`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_add_bias_`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_add_relu`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_backward_bias`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_backward_input`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_backward_weight`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_forward_out`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_relu`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_transpose`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_transpose_backward_input`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_transpose_backward_weight`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_convolution_transpose_forward`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_depthwise_convolution`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_depthwise_convolution_backward_input`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`miopen_depthwise_convolution_backward_weight`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`raw_miopen_convolution_add_relu_out`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`raw_miopen_convolution_backward_input_out`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`raw_miopen_convolution_backward_input_out_32bit`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`raw_miopen_convolution_backward_weight_out`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`raw_miopen_convolution_backward_weight_out_32bit`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`raw_miopen_convolution_forward_out`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`raw_miopen_convolution_forward_out_32bit`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`self_or_new_memory_format`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`setConvolutionParams`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`split_batch_dim_to_32bit_out`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)

### Includes

- **`ATen/Config.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/Functions.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/cuda/CUDAConfig.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/hip/EmptyTensor.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/miopen/Descriptors.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/miopen/Types.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/miopen/Utils.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/miopen/miopen-wrapper.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/native/ConvUtils.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/native/utils/ParamsHash.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/empty.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/empty_like.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/empty_native.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/miopen_convolution_add_relu_native.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/miopen_convolution_native.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/miopen_convolution_relu_native.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/miopen_convolution_transpose_native.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/miopen_depthwise_convolution_native.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/squeeze.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/sum.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`ATen/ops/zeros_like.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`algorithm`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`c10/hip/HIPCachingAllocator.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`c10/util/irange.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`functional`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`iterator`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`memory`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`mutex`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`stdint.h`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`unordered_map`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)

### Namespaces

- **`at`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)
- **`native`**: [Conv_miopen.cpp_docs.md](./Conv_miopen.cpp_docs.md)


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
