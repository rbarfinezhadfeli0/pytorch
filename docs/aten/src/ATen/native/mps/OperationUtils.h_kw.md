# Keyword Index: `aten/src/ATen/native/mps/OperationUtils.h`

## File Information

- **Original File**: [aten/src/ATen/native/mps/OperationUtils.h](../../../../../../aten/src/ATen/native/mps/OperationUtils.h)
- **Documentation**: [`OperationUtils.h_docs.md`](./OperationUtils.h_docs.md)
- **Folder**: `aten/src/ATen/native/mps`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`CacheEntry`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSBinaryCachedGraph`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSBinaryGradCachedGraph`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSCachedGraph`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSCachedKernel`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSGraphCache`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSKernelCache`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSScalar`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSUnaryCachedGraph`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MPSUnaryGradCachedGraph`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`Placeholder`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`to`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)

### Functions

- **`bind_iter_tensors`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`constexpr`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`getMPSDataType`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`getMPSScalarType`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`getMPSTypeString`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`isIntermediate`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`is_dense_in_storage`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`iter_tensor_offset`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`mtl_dispatch1DJob`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`mtl_setArg`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`mtl_setArgs`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`mtl_setBuffer`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`mtl_setBytes`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`needsGather`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`runMPSGraph`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`scalarToMetalTypeString`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`supportedFloatingOrComplexType`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`supportedFloatingType`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)

### Includes

- **`ATen/Functions.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/NativeFunctions.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/Tensor.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/TensorIterator.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/Utils.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/mps/MPSStream.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/native/mps/MetalShaderLibrary.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/native/mps/TensorFactory.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/ops/empty.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/ops/empty_like.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/ops/zeros.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`ATen/ops/zeros_like.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`MetalPerformanceShaders/MetalPerformanceShaders.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`c10/core/ScalarType.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`initializer_list`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`torch/library.h`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`unordered_map`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)

### Namespaces

- **`at`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)
- **`detail`**: [OperationUtils.h_docs.md](./OperationUtils.h_docs.md)


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
