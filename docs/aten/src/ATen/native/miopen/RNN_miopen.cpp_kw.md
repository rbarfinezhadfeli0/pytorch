# Keyword Index: `aten/src/ATen/native/miopen/RNN_miopen.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/miopen/RNN_miopen.cpp](../../../../../../aten/src/ATen/native/miopen/RNN_miopen.cpp)
- **Documentation**: [`RNN_miopen.cpp_docs.md`](./RNN_miopen.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/miopen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`DropoutState`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`RNNDescriptorParams`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`RNNDescriptors`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`RNNParams`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`TensorDescriptorListParams`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)

### Functions

- **`_copyParams`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`_copyParams_and_permute`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`_num_linear_layers`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`_viewOrCopyParams`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`_viewParams`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`descriptor`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`descriptorWithDropout`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`get_num_weights`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`if`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`is_input_packed`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`lstm_miopen`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`lstm_packed_miopen`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`num_directions`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`pack_hidden`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`permute_wei_for_miopen`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`set`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`set_algo`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`set_bidirectional`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`set_dropout`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`set_mode`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)

### Includes

- **`ATen/Config.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/Functions.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/InitialTensorOptions.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/MatrixRef.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/cuda/CUDAConfig.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/miopen/Descriptors.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/miopen/Types.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/miopen/Utils.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/miopen/miopen-wrapper.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/native/RNN.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/ops/cat.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/ops/empty.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/ops/miopen_rnn.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/ops/miopen_rnn_backward_native.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/ops/miopen_rnn_native.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`ATen/ops/zeros_like.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`algorithm`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`c10/hip/HIPCachingAllocator.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`c10/util/Exception.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`c10/util/irange.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`functional`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`iterator`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`memory`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`mutex`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`rocrand/rocrand_xorwow.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`sstream`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`stdint.h`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)
- **`unordered_map`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)

### Namespaces

- **`at`**: [RNN_miopen.cpp_docs.md](./RNN_miopen.cpp_docs.md)


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
