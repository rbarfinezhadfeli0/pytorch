# Index: `aten/src/ATen/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `aten/src/ATen/`

## Subfolders

- [`benchmarks/`](./benchmarks/index.md) - Performance benchmarks
- [`core/`](./core/index.md) - core module
- [`cpu/`](./cpu/index.md) - cpu module
- [`cuda/`](./cuda/index.md) - cuda module
- [`cudnn/`](./cudnn/index.md) - cudnn module
- [`detail/`](./detail/index.md) - detail module
- [`functorch/`](./functorch/index.md) - functorch module
- [`hip/`](./hip/index.md) - hip module
- [`metal/`](./metal/index.md) - metal module
- [`miopen/`](./miopen/index.md) - miopen module
- [`mkl/`](./mkl/index.md) - mkl module
- [`mps/`](./mps/index.md) - mps module
- [`native/`](./native/index.md) - native module
- [`nnapi/`](./nnapi/index.md) - nnapi module
- [`ops/`](./ops/index.md) - ops module
- [`quantized/`](./quantized/index.md) - quantized module
- [`templates/`](./templates/index.md) - templates module
- [`test/`](./test/index.md) - Test suites
- [`vulkan/`](./vulkan/index.md) - vulkan module
- [`xpu/`](./xpu/index.md) - xpu module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`ATen.h`](../../../../aten/src/ATen/ATen.h) | Source code | [docs](./ATen.h_docs.md) | [keywords](./ATen.h_kw.md) |
| [`AccumulateType.cpp`](../../../../aten/src/ATen/AccumulateType.cpp) | Source code | [docs](./AccumulateType.cpp_docs.md) | [keywords](./AccumulateType.cpp_kw.md) |
| [`AccumulateType.h`](../../../../aten/src/ATen/AccumulateType.h) | Source code | [docs](./AccumulateType.h_docs.md) | [keywords](./AccumulateType.h_kw.md) |
| [`ArrayRef.h`](../../../../aten/src/ATen/ArrayRef.h) | Source code | [docs](./ArrayRef.h_docs.md) | [keywords](./ArrayRef.h_kw.md) |
| [`Backend.h`](../../../../aten/src/ATen/Backend.h) | Source code | [docs](./Backend.h_docs.md) | [keywords](./Backend.h_kw.md) |
| [`Backtrace.h`](../../../../aten/src/ATen/Backtrace.h) | Source code | [docs](./Backtrace.h_docs.md) | [keywords](./Backtrace.h_kw.md) |
| [`BlasBackend.h`](../../../../aten/src/ATen/BlasBackend.h) | Source code | [docs](./BlasBackend.h_docs.md) | [keywords](./BlasBackend.h_kw.md) |
| [`CMakeLists.txt`](../../../../aten/src/ATen/CMakeLists.txt) | Source code | [docs](./CMakeLists.txt_docs.md) | [keywords](./CMakeLists.txt_kw.md) |
| [`CPUApplyUtils.h`](../../../../aten/src/ATen/CPUApplyUtils.h) | Source code | [docs](./CPUApplyUtils.h_docs.md) | [keywords](./CPUApplyUtils.h_kw.md) |
| [`CPUFixedAllocator.h`](../../../../aten/src/ATen/CPUFixedAllocator.h) | Source code | [docs](./CPUFixedAllocator.h_docs.md) | [keywords](./CPUFixedAllocator.h_kw.md) |
| [`CPUGeneratorImpl.cpp`](../../../../aten/src/ATen/CPUGeneratorImpl.cpp) | Source code | [docs](./CPUGeneratorImpl.cpp_docs.md) | [keywords](./CPUGeneratorImpl.cpp_kw.md) |
| [`CPUGeneratorImpl.h`](../../../../aten/src/ATen/CPUGeneratorImpl.h) | Source code | [docs](./CPUGeneratorImpl.h_docs.md) | [keywords](./CPUGeneratorImpl.h_kw.md) |
| [`CachedTensorUtils.cpp`](../../../../aten/src/ATen/CachedTensorUtils.cpp) | Source code | [docs](./CachedTensorUtils.cpp_docs.md) | [keywords](./CachedTensorUtils.cpp_kw.md) |
| [`CachedTensorUtils.h`](../../../../aten/src/ATen/CachedTensorUtils.h) | Source code | [docs](./CachedTensorUtils.h_docs.md) | [keywords](./CachedTensorUtils.h_kw.md) |
| [`CollapseDims.h`](../../../../aten/src/ATen/CollapseDims.h) | Source code | [docs](./CollapseDims.h_docs.md) | [keywords](./CollapseDims.h_kw.md) |
| [`ConjugateFallback.cpp`](../../../../aten/src/ATen/ConjugateFallback.cpp) | Source code | [docs](./ConjugateFallback.cpp_docs.md) | [keywords](./ConjugateFallback.cpp_kw.md) |
| [`Context.cpp`](../../../../aten/src/ATen/Context.cpp) | Source code | [docs](./Context.cpp_docs.md) | [keywords](./Context.cpp_kw.md) |
| [`Context.h`](../../../../aten/src/ATen/Context.h) | Source code | [docs](./Context.h_docs.md) | [keywords](./Context.h_kw.md) |
| [`DLConvertor.cpp`](../../../../aten/src/ATen/DLConvertor.cpp) | Source code | [docs](./DLConvertor.cpp_docs.md) | [keywords](./DLConvertor.cpp_kw.md) |
| [`DLConvertor.h`](../../../../aten/src/ATen/DLConvertor.h) | Source code | [docs](./DLConvertor.h_docs.md) | [keywords](./DLConvertor.h_kw.md) |
| [`DTensorState.cpp`](../../../../aten/src/ATen/DTensorState.cpp) | Source code | [docs](./DTensorState.cpp_docs.md) | [keywords](./DTensorState.cpp_kw.md) |
| [`DTensorState.h`](../../../../aten/src/ATen/DTensorState.h) | Source code | [docs](./DTensorState.h_docs.md) | [keywords](./DTensorState.h_kw.md) |
| [`Device.h`](../../../../aten/src/ATen/Device.h) | Source code | [docs](./Device.h_docs.md) | [keywords](./Device.h_kw.md) |
| [`DeviceAccelerator.cpp`](../../../../aten/src/ATen/DeviceAccelerator.cpp) | Source code | [docs](./DeviceAccelerator.cpp_docs.md) | [keywords](./DeviceAccelerator.cpp_kw.md) |
| [`DeviceAccelerator.h`](../../../../aten/src/ATen/DeviceAccelerator.h) | Source code | [docs](./DeviceAccelerator.h_docs.md) | [keywords](./DeviceAccelerator.h_kw.md) |
| [`DeviceGuard.h`](../../../../aten/src/ATen/DeviceGuard.h) | Source code | [docs](./DeviceGuard.h_docs.md) | [keywords](./DeviceGuard.h_kw.md) |
| [`DimVector.h`](../../../../aten/src/ATen/DimVector.h) | Source code | [docs](./DimVector.h_docs.md) | [keywords](./DimVector.h_kw.md) |
| [`Dimname.h`](../../../../aten/src/ATen/Dimname.h) | Source code | [docs](./Dimname.h_docs.md) | [keywords](./Dimname.h_kw.md) |
| [`Dispatch.cpp`](../../../../aten/src/ATen/Dispatch.cpp) | Source code | [docs](./Dispatch.cpp_docs.md) | [keywords](./Dispatch.cpp_kw.md) |
| [`Dispatch.h`](../../../../aten/src/ATen/Dispatch.h) | Source code | [docs](./Dispatch.h_docs.md) | [keywords](./Dispatch.h_kw.md) |
| [`Dispatch_v2.h`](../../../../aten/src/ATen/Dispatch_v2.h) | Source code | [docs](./Dispatch_v2.h_docs.md) | [keywords](./Dispatch_v2.h_kw.md) |
| [`DynamicLibrary.cpp`](../../../../aten/src/ATen/DynamicLibrary.cpp) | Source code | [docs](./DynamicLibrary.cpp_docs.md) | [keywords](./DynamicLibrary.cpp_kw.md) |
| [`DynamicLibrary.h`](../../../../aten/src/ATen/DynamicLibrary.h) | Source code | [docs](./DynamicLibrary.h_docs.md) | [keywords](./DynamicLibrary.h_kw.md) |
| [`EmptyTensor.cpp`](../../../../aten/src/ATen/EmptyTensor.cpp) | Source code | [docs](./EmptyTensor.cpp_docs.md) | [keywords](./EmptyTensor.cpp_kw.md) |
| [`EmptyTensor.h`](../../../../aten/src/ATen/EmptyTensor.h) | Source code | [docs](./EmptyTensor.h_docs.md) | [keywords](./EmptyTensor.h_kw.md) |
| [`ExpandBase.h`](../../../../aten/src/ATen/ExpandBase.h) | Source code | [docs](./ExpandBase.h_docs.md) | [keywords](./ExpandBase.h_kw.md) |
| [`ExpandUtils.cpp`](../../../../aten/src/ATen/ExpandUtils.cpp) | Source code | [docs](./ExpandUtils.cpp_docs.md) | [keywords](./ExpandUtils.cpp_kw.md) |
| [`ExpandUtils.h`](../../../../aten/src/ATen/ExpandUtils.h) | Source code | [docs](./ExpandUtils.h_docs.md) | [keywords](./ExpandUtils.h_kw.md) |
| [`Formatting.h`](../../../../aten/src/ATen/Formatting.h) | Source code | [docs](./Formatting.h_docs.md) | [keywords](./Formatting.h_kw.md) |
| [`FuncTorchTLS.cpp`](../../../../aten/src/ATen/FuncTorchTLS.cpp) | Source code | [docs](./FuncTorchTLS.cpp_docs.md) | [keywords](./FuncTorchTLS.cpp_kw.md) |
| [`FuncTorchTLS.h`](../../../../aten/src/ATen/FuncTorchTLS.h) | Source code | [docs](./FuncTorchTLS.h_docs.md) | [keywords](./FuncTorchTLS.h_kw.md) |
| [`FunctionalInverses.cpp`](../../../../aten/src/ATen/FunctionalInverses.cpp) | Source code | [docs](./FunctionalInverses.cpp_docs.md) | [keywords](./FunctionalInverses.cpp_kw.md) |
| [`FunctionalStorageImpl.cpp`](../../../../aten/src/ATen/FunctionalStorageImpl.cpp) | Source code | [docs](./FunctionalStorageImpl.cpp_docs.md) | [keywords](./FunctionalStorageImpl.cpp_kw.md) |
| [`FunctionalStorageImpl.h`](../../../../aten/src/ATen/FunctionalStorageImpl.h) | Source code | [docs](./FunctionalStorageImpl.h_docs.md) | [keywords](./FunctionalStorageImpl.h_kw.md) |
| [`FunctionalTensorWrapper.cpp`](../../../../aten/src/ATen/FunctionalTensorWrapper.cpp) | Source code | [docs](./FunctionalTensorWrapper.cpp_docs.md) | [keywords](./FunctionalTensorWrapper.cpp_kw.md) |
| [`FunctionalTensorWrapper.h`](../../../../aten/src/ATen/FunctionalTensorWrapper.h) | Source code | [docs](./FunctionalTensorWrapper.h_docs.md) | [keywords](./FunctionalTensorWrapper.h_kw.md) |
| [`FunctionalizeFallbackKernel.cpp`](../../../../aten/src/ATen/FunctionalizeFallbackKernel.cpp) | Source code | [docs](./FunctionalizeFallbackKernel.cpp_docs.md) | [keywords](./FunctionalizeFallbackKernel.cpp_kw.md) |
| [`FunctionalizeFallbackKernel.h`](../../../../aten/src/ATen/FunctionalizeFallbackKernel.h) | Source code | [docs](./FunctionalizeFallbackKernel.h_docs.md) | [keywords](./FunctionalizeFallbackKernel.h_kw.md) |
| [`Generator.h`](../../../../aten/src/ATen/Generator.h) | Source code | [docs](./Generator.h_docs.md) | [keywords](./Generator.h_kw.md) |
| [`InferSize.h`](../../../../aten/src/ATen/InferSize.h) | Source code | [docs](./InferSize.h_docs.md) | [keywords](./InferSize.h_kw.md) |
| [`InitialTensorOptions.h`](../../../../aten/src/ATen/InitialTensorOptions.h) | Source code | [docs](./InitialTensorOptions.h_docs.md) | [keywords](./InitialTensorOptions.h_kw.md) |
| [`Layout.h`](../../../../aten/src/ATen/Layout.h) | Source code | [docs](./Layout.h_docs.md) | [keywords](./Layout.h_kw.md) |
| [`LegacyBatchedFallback.cpp`](../../../../aten/src/ATen/LegacyBatchedFallback.cpp) | Source code | [docs](./LegacyBatchedFallback.cpp_docs.md) | [keywords](./LegacyBatchedFallback.cpp_kw.md) |
| [`LegacyBatchedFallback.h`](../../../../aten/src/ATen/LegacyBatchedFallback.h) | Source code | [docs](./LegacyBatchedFallback.h_docs.md) | [keywords](./LegacyBatchedFallback.h_kw.md) |
| [`LegacyBatchedTensorImpl.cpp`](../../../../aten/src/ATen/LegacyBatchedTensorImpl.cpp) | Source code | [docs](./LegacyBatchedTensorImpl.cpp_docs.md) | [keywords](./LegacyBatchedTensorImpl.cpp_kw.md) |
| [`LegacyBatchedTensorImpl.h`](../../../../aten/src/ATen/LegacyBatchedTensorImpl.h) | Source code | [docs](./LegacyBatchedTensorImpl.h_docs.md) | [keywords](./LegacyBatchedTensorImpl.h_kw.md) |
| [`LegacyBatchingRegistrations.cpp`](../../../../aten/src/ATen/LegacyBatchingRegistrations.cpp) | Source code | [docs](./LegacyBatchingRegistrations.cpp_docs.md) | [keywords](./LegacyBatchingRegistrations.cpp_kw.md) |
| [`LegacyVmapMode.cpp`](../../../../aten/src/ATen/LegacyVmapMode.cpp) | Source code | [docs](./LegacyVmapMode.cpp_docs.md) | [keywords](./LegacyVmapMode.cpp_kw.md) |
| [`LegacyVmapMode.h`](../../../../aten/src/ATen/LegacyVmapMode.h) | Source code | [docs](./LegacyVmapMode.h_docs.md) | [keywords](./LegacyVmapMode.h_kw.md) |
| [`LegacyVmapTransforms.cpp`](../../../../aten/src/ATen/LegacyVmapTransforms.cpp) | Source code | [docs](./LegacyVmapTransforms.cpp_docs.md) | [keywords](./LegacyVmapTransforms.cpp_kw.md) |
| [`LegacyVmapTransforms.h`](../../../../aten/src/ATen/LegacyVmapTransforms.h) | Source code | [docs](./LegacyVmapTransforms.h_docs.md) | [keywords](./LegacyVmapTransforms.h_kw.md) |
| [`LinalgBackend.h`](../../../../aten/src/ATen/LinalgBackend.h) | Source code | [docs](./LinalgBackend.h_docs.md) | [keywords](./LinalgBackend.h_kw.md) |
| [`MapAllocator.cpp`](../../../../aten/src/ATen/MapAllocator.cpp) | Source code | [docs](./MapAllocator.cpp_docs.md) | [keywords](./MapAllocator.cpp_kw.md) |
| [`MapAllocator.h`](../../../../aten/src/ATen/MapAllocator.h) | Source code | [docs](./MapAllocator.h_docs.md) | [keywords](./MapAllocator.h_kw.md) |
| [`MatrixRef.h`](../../../../aten/src/ATen/MatrixRef.h) | Source code | [docs](./MatrixRef.h_docs.md) | [keywords](./MatrixRef.h_kw.md) |
| [`MemoryOverlap.cpp`](../../../../aten/src/ATen/MemoryOverlap.cpp) | Source code | [docs](./MemoryOverlap.cpp_docs.md) | [keywords](./MemoryOverlap.cpp_kw.md) |
| [`MemoryOverlap.h`](../../../../aten/src/ATen/MemoryOverlap.h) | Source code | [docs](./MemoryOverlap.h_docs.md) | [keywords](./MemoryOverlap.h_kw.md) |
| [`NamedTensor.h`](../../../../aten/src/ATen/NamedTensor.h) | Source code | [docs](./NamedTensor.h_docs.md) | [keywords](./NamedTensor.h_kw.md) |
| [`NamedTensorUtils.cpp`](../../../../aten/src/ATen/NamedTensorUtils.cpp) | Source code | [docs](./NamedTensorUtils.cpp_docs.md) | [keywords](./NamedTensorUtils.cpp_kw.md) |
| [`NamedTensorUtils.h`](../../../../aten/src/ATen/NamedTensorUtils.h) | Source code | [docs](./NamedTensorUtils.h_docs.md) | [keywords](./NamedTensorUtils.h_kw.md) |
| [`NestedTensorImpl.cpp`](../../../../aten/src/ATen/NestedTensorImpl.cpp) | Source code | [docs](./NestedTensorImpl.cpp_docs.md) | [keywords](./NestedTensorImpl.cpp_kw.md) |
| [`NestedTensorImpl.h`](../../../../aten/src/ATen/NestedTensorImpl.h) | Source code | [docs](./NestedTensorImpl.h_docs.md) | [keywords](./NestedTensorImpl.h_kw.md) |
| [`NumericUtils.h`](../../../../aten/src/ATen/NumericUtils.h) | Source code | [docs](./NumericUtils.h_docs.md) | [keywords](./NumericUtils.h_kw.md) |
| [`OpMathType.h`](../../../../aten/src/ATen/OpMathType.h) | Source code | [docs](./OpMathType.h_docs.md) | [keywords](./OpMathType.h_kw.md) |
| [`OpaqueTensorImpl.h`](../../../../aten/src/ATen/OpaqueTensorImpl.h) | Source code | [docs](./OpaqueTensorImpl.h_docs.md) | [keywords](./OpaqueTensorImpl.h_kw.md) |
| [`PTThreadPool.h`](../../../../aten/src/ATen/PTThreadPool.h) | Source code | [docs](./PTThreadPool.h_docs.md) | [keywords](./PTThreadPool.h_kw.md) |
| [`PadNd.h`](../../../../aten/src/ATen/PadNd.h) | Source code | [docs](./PadNd.h_docs.md) | [keywords](./PadNd.h_kw.md) |
| [`Parallel-inl.h`](../../../../aten/src/ATen/Parallel-inl.h) | Source code | [docs](./Parallel-inl.h_docs.md) | [keywords](./Parallel-inl.h_kw.md) |
| [`Parallel.h`](../../../../aten/src/ATen/Parallel.h) | Source code | [docs](./Parallel.h_docs.md) | [keywords](./Parallel.h_kw.md) |
| [`ParallelCommon.cpp`](../../../../aten/src/ATen/ParallelCommon.cpp) | Source code | [docs](./ParallelCommon.cpp_docs.md) | [keywords](./ParallelCommon.cpp_kw.md) |
| [`ParallelFuture.h`](../../../../aten/src/ATen/ParallelFuture.h) | Source code | [docs](./ParallelFuture.h_docs.md) | [keywords](./ParallelFuture.h_kw.md) |
| [`ParallelNative.cpp`](../../../../aten/src/ATen/ParallelNative.cpp) | Source code | [docs](./ParallelNative.cpp_docs.md) | [keywords](./ParallelNative.cpp_kw.md) |
| [`ParallelNative.h`](../../../../aten/src/ATen/ParallelNative.h) | Source code | [docs](./ParallelNative.h_docs.md) | [keywords](./ParallelNative.h_kw.md) |
| [`ParallelOpenMP.cpp`](../../../../aten/src/ATen/ParallelOpenMP.cpp) | Source code | [docs](./ParallelOpenMP.cpp_docs.md) | [keywords](./ParallelOpenMP.cpp_kw.md) |
| [`ParallelOpenMP.h`](../../../../aten/src/ATen/ParallelOpenMP.h) | Source code | [docs](./ParallelOpenMP.h_docs.md) | [keywords](./ParallelOpenMP.h_kw.md) |
| [`ParallelThreadPoolNative.cpp`](../../../../aten/src/ATen/ParallelThreadPoolNative.cpp) | Source code | [docs](./ParallelThreadPoolNative.cpp_docs.md) | [keywords](./ParallelThreadPoolNative.cpp_kw.md) |
| [`PythonTorchFunctionTLS.cpp`](../../../../aten/src/ATen/PythonTorchFunctionTLS.cpp) | Source code | [docs](./PythonTorchFunctionTLS.cpp_docs.md) | [keywords](./PythonTorchFunctionTLS.cpp_kw.md) |
| [`PythonTorchFunctionTLS.h`](../../../../aten/src/ATen/PythonTorchFunctionTLS.h) | Source code | [docs](./PythonTorchFunctionTLS.h_docs.md) | [keywords](./PythonTorchFunctionTLS.h_kw.md) |
| [`ROCmFABackend.h`](../../../../aten/src/ATen/ROCmFABackend.h) | Source code | [docs](./ROCmFABackend.h_docs.md) | [keywords](./ROCmFABackend.h_kw.md) |
| [`SDPBackend.h`](../../../../aten/src/ATen/SDPBackend.h) | Source code | [docs](./SDPBackend.h_docs.md) | [keywords](./SDPBackend.h_kw.md) |
| [`SavedTensorHooks.cpp`](../../../../aten/src/ATen/SavedTensorHooks.cpp) | Source code | [docs](./SavedTensorHooks.cpp_docs.md) | [keywords](./SavedTensorHooks.cpp_kw.md) |
| [`SavedTensorHooks.h`](../../../../aten/src/ATen/SavedTensorHooks.h) | Source code | [docs](./SavedTensorHooks.h_docs.md) | [keywords](./SavedTensorHooks.h_kw.md) |
| [`Scalar.h`](../../../../aten/src/ATen/Scalar.h) | Source code | [docs](./Scalar.h_docs.md) | [keywords](./Scalar.h_kw.md) |
| [`ScalarOps.cpp`](../../../../aten/src/ATen/ScalarOps.cpp) | Source code | [docs](./ScalarOps.cpp_docs.md) | [keywords](./ScalarOps.cpp_kw.md) |
| [`ScalarOps.h`](../../../../aten/src/ATen/ScalarOps.h) | Source code | [docs](./ScalarOps.h_docs.md) | [keywords](./ScalarOps.h_kw.md) |
| [`ScalarType.h`](../../../../aten/src/ATen/ScalarType.h) | Source code | [docs](./ScalarType.h_docs.md) | [keywords](./ScalarType.h_kw.md) |
| [`SequenceNumber.cpp`](../../../../aten/src/ATen/SequenceNumber.cpp) | Source code | [docs](./SequenceNumber.cpp_docs.md) | [keywords](./SequenceNumber.cpp_kw.md) |
| [`SequenceNumber.h`](../../../../aten/src/ATen/SequenceNumber.h) | Source code | [docs](./SequenceNumber.h_docs.md) | [keywords](./SequenceNumber.h_kw.md) |
| [`SmallVector.h`](../../../../aten/src/ATen/SmallVector.h) | Source code | [docs](./SmallVector.h_docs.md) | [keywords](./SmallVector.h_kw.md) |
| [`SparseCsrTensorImpl.cpp`](../../../../aten/src/ATen/SparseCsrTensorImpl.cpp) | Source code | [docs](./SparseCsrTensorImpl.cpp_docs.md) | [keywords](./SparseCsrTensorImpl.cpp_kw.md) |
| [`SparseCsrTensorImpl.h`](../../../../aten/src/ATen/SparseCsrTensorImpl.h) | Source code | [docs](./SparseCsrTensorImpl.h_docs.md) | [keywords](./SparseCsrTensorImpl.h_kw.md) |
| [`SparseCsrTensorUtils.h`](../../../../aten/src/ATen/SparseCsrTensorUtils.h) | Source code | [docs](./SparseCsrTensorUtils.h_docs.md) | [keywords](./SparseCsrTensorUtils.h_kw.md) |
| [`SparseTensorImpl.cpp`](../../../../aten/src/ATen/SparseTensorImpl.cpp) | Source code | [docs](./SparseTensorImpl.cpp_docs.md) | [keywords](./SparseTensorImpl.cpp_kw.md) |
| [`SparseTensorImpl.h`](../../../../aten/src/ATen/SparseTensorImpl.h) | Source code | [docs](./SparseTensorImpl.h_docs.md) | [keywords](./SparseTensorImpl.h_kw.md) |
| [`Storage.h`](../../../../aten/src/ATen/Storage.h) | Source code | [docs](./Storage.h_docs.md) | [keywords](./Storage.h_kw.md) |
| [`StorageUtils.cpp`](../../../../aten/src/ATen/StorageUtils.cpp) | Source code | [docs](./StorageUtils.cpp_docs.md) | [keywords](./StorageUtils.cpp_kw.md) |
| [`StorageUtils.h`](../../../../aten/src/ATen/StorageUtils.h) | Source code | [docs](./StorageUtils.h_docs.md) | [keywords](./StorageUtils.h_kw.md) |
| [`Tensor.h`](../../../../aten/src/ATen/Tensor.h) | Source code | [docs](./Tensor.h_docs.md) | [keywords](./Tensor.h_kw.md) |
| [`TensorAccessor.h`](../../../../aten/src/ATen/TensorAccessor.h) | Source code | [docs](./TensorAccessor.h_docs.md) | [keywords](./TensorAccessor.h_kw.md) |
| [`TensorGeometry.cpp`](../../../../aten/src/ATen/TensorGeometry.cpp) | Source code | [docs](./TensorGeometry.cpp_docs.md) | [keywords](./TensorGeometry.cpp_kw.md) |
| [`TensorGeometry.h`](../../../../aten/src/ATen/TensorGeometry.h) | Source code | [docs](./TensorGeometry.h_docs.md) | [keywords](./TensorGeometry.h_kw.md) |
| [`TensorIndexing.cpp`](../../../../aten/src/ATen/TensorIndexing.cpp) | Source code | [docs](./TensorIndexing.cpp_docs.md) | [keywords](./TensorIndexing.cpp_kw.md) |
| [`TensorIndexing.h`](../../../../aten/src/ATen/TensorIndexing.h) | Source code | [docs](./TensorIndexing.h_docs.md) | [keywords](./TensorIndexing.h_kw.md) |
| [`TensorIterator.cpp`](../../../../aten/src/ATen/TensorIterator.cpp) | Source code | [docs](./TensorIterator.cpp_docs.md) | [keywords](./TensorIterator.cpp_kw.md) |
| [`TensorIterator.h`](../../../../aten/src/ATen/TensorIterator.h) | Source code | [docs](./TensorIterator.h_docs.md) | [keywords](./TensorIterator.h_kw.md) |
| [`TensorIteratorInternal.h`](../../../../aten/src/ATen/TensorIteratorInternal.h) | Source code | [docs](./TensorIteratorInternal.h_docs.md) | [keywords](./TensorIteratorInternal.h_kw.md) |
| [`TensorMeta.cpp`](../../../../aten/src/ATen/TensorMeta.cpp) | Source code | [docs](./TensorMeta.cpp_docs.md) | [keywords](./TensorMeta.cpp_kw.md) |
| [`TensorMeta.h`](../../../../aten/src/ATen/TensorMeta.h) | Source code | [docs](./TensorMeta.h_docs.md) | [keywords](./TensorMeta.h_kw.md) |
| [`TensorNames.cpp`](../../../../aten/src/ATen/TensorNames.cpp) | Source code | [docs](./TensorNames.cpp_docs.md) | [keywords](./TensorNames.cpp_kw.md) |
| [`TensorNames.h`](../../../../aten/src/ATen/TensorNames.h) | Source code | [docs](./TensorNames.h_docs.md) | [keywords](./TensorNames.h_kw.md) |
| [`TensorOperators.h`](../../../../aten/src/ATen/TensorOperators.h) | Source code | [docs](./TensorOperators.h_docs.md) | [keywords](./TensorOperators.h_kw.md) |
| [`TensorOptions.h`](../../../../aten/src/ATen/TensorOptions.h) | Source code | [docs](./TensorOptions.h_docs.md) | [keywords](./TensorOptions.h_kw.md) |
| [`TensorSubclassLikeUtils.h`](../../../../aten/src/ATen/TensorSubclassLikeUtils.h) | Source code | [docs](./TensorSubclassLikeUtils.h_docs.md) | [keywords](./TensorSubclassLikeUtils.h_kw.md) |
| [`TensorUtils.cpp`](../../../../aten/src/ATen/TensorUtils.cpp) | Source code | [docs](./TensorUtils.cpp_docs.md) | [keywords](./TensorUtils.cpp_kw.md) |
| [`TensorUtils.h`](../../../../aten/src/ATen/TensorUtils.h) | Source code | [docs](./TensorUtils.h_docs.md) | [keywords](./TensorUtils.h_kw.md) |
| [`ThreadLocalPythonObjects.cpp`](../../../../aten/src/ATen/ThreadLocalPythonObjects.cpp) | Source code | [docs](./ThreadLocalPythonObjects.cpp_docs.md) | [keywords](./ThreadLocalPythonObjects.cpp_kw.md) |
| [`ThreadLocalPythonObjects.h`](../../../../aten/src/ATen/ThreadLocalPythonObjects.h) | Source code | [docs](./ThreadLocalPythonObjects.h_docs.md) | [keywords](./ThreadLocalPythonObjects.h_kw.md) |
| [`ThreadLocalState.cpp`](../../../../aten/src/ATen/ThreadLocalState.cpp) | Source code | [docs](./ThreadLocalState.cpp_docs.md) | [keywords](./ThreadLocalState.cpp_kw.md) |
| [`ThreadLocalState.h`](../../../../aten/src/ATen/ThreadLocalState.h) | Source code | [docs](./ThreadLocalState.h_docs.md) | [keywords](./ThreadLocalState.h_kw.md) |
| [`TracerMode.h`](../../../../aten/src/ATen/TracerMode.h) | Source code | [docs](./TracerMode.h_docs.md) | [keywords](./TracerMode.h_kw.md) |
| [`TypeDefault.h`](../../../../aten/src/ATen/TypeDefault.h) | Source code | [docs](./TypeDefault.h_docs.md) | [keywords](./TypeDefault.h_kw.md) |
| [`Utils.cpp`](../../../../aten/src/ATen/Utils.cpp) | Source code | [docs](./Utils.cpp_docs.md) | [keywords](./Utils.cpp_kw.md) |
| [`Utils.h`](../../../../aten/src/ATen/Utils.h) | Source code | [docs](./Utils.h_docs.md) | [keywords](./Utils.h_kw.md) |
| [`Version.cpp`](../../../../aten/src/ATen/Version.cpp) | Source code | [docs](./Version.cpp_docs.md) | [keywords](./Version.cpp_kw.md) |
| [`Version.h`](../../../../aten/src/ATen/Version.h) | Source code | [docs](./Version.h_docs.md) | [keywords](./Version.h_kw.md) |
| [`VmapModeRegistrations.cpp`](../../../../aten/src/ATen/VmapModeRegistrations.cpp) | Source code | [docs](./VmapModeRegistrations.cpp_docs.md) | [keywords](./VmapModeRegistrations.cpp_kw.md) |
| [`WrapDimUtils.h`](../../../../aten/src/ATen/WrapDimUtils.h) | Source code | [docs](./WrapDimUtils.h_docs.md) | [keywords](./WrapDimUtils.h_kw.md) |
| [`WrapDimUtilsMulti.h`](../../../../aten/src/ATen/WrapDimUtilsMulti.h) | Source code | [docs](./WrapDimUtilsMulti.h_docs.md) | [keywords](./WrapDimUtilsMulti.h_kw.md) |
| [`ZeroTensorFallback.cpp`](../../../../aten/src/ATen/ZeroTensorFallback.cpp) | Source code | [docs](./ZeroTensorFallback.cpp_docs.md) | [keywords](./ZeroTensorFallback.cpp_kw.md) |
| [`autocast_mode.cpp`](../../../../aten/src/ATen/autocast_mode.cpp) | Source code | [docs](./autocast_mode.cpp_docs.md) | [keywords](./autocast_mode.cpp_kw.md) |
| [`autocast_mode.h`](../../../../aten/src/ATen/autocast_mode.h) | Source code | [docs](./autocast_mode.h_docs.md) | [keywords](./autocast_mode.h_kw.md) |
| [`ceil_div.h`](../../../../aten/src/ATen/ceil_div.h) | Source code | [docs](./ceil_div.h_docs.md) | [keywords](./ceil_div.h_kw.md) |
| [`code_template.h`](../../../../aten/src/ATen/code_template.h) | Source code | [docs](./code_template.h_docs.md) | [keywords](./code_template.h_kw.md) |
| [`cpp_custom_type_hack.h`](../../../../aten/src/ATen/cpp_custom_type_hack.h) | Source code | [docs](./cpp_custom_type_hack.h_docs.md) | [keywords](./cpp_custom_type_hack.h_kw.md) |
| [`div_rtn.h`](../../../../aten/src/ATen/div_rtn.h) | Source code | [docs](./div_rtn.h_docs.md) | [keywords](./div_rtn.h_kw.md) |
| [`dlpack.h`](../../../../aten/src/ATen/dlpack.h) | Source code | [docs](./dlpack.h_docs.md) | [keywords](./dlpack.h_kw.md) |
| [`jit_macros.h`](../../../../aten/src/ATen/jit_macros.h) | Source code | [docs](./jit_macros.h_docs.md) | [keywords](./jit_macros.h_kw.md) |
| [`jiterator_macros.h`](../../../../aten/src/ATen/jiterator_macros.h) | Source code | [docs](./jiterator_macros.h_docs.md) | [keywords](./jiterator_macros.h_kw.md) |
| [`record_function.cpp`](../../../../aten/src/ATen/record_function.cpp) | Source code | [docs](./record_function.cpp_docs.md) | [keywords](./record_function.cpp_kw.md) |
| [`record_function.h`](../../../../aten/src/ATen/record_function.h) | Source code | [docs](./record_function.h_docs.md) | [keywords](./record_function.h_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
