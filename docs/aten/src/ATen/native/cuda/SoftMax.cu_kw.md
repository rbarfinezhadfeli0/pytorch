# Keyword Index: `aten/src/ATen/native/cuda/SoftMax.cu`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/SoftMax.cu](../../../../../../aten/src/ATen/native/cuda/SoftMax.cu)
- **Documentation**: [`SoftMax.cu_docs.md`](./SoftMax.cu_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Add`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`AddFloat`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`Epilogue`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`EpilogueWithMul`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`LogSoftMaxBackwardEpilogue`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`LogSoftMaxForwardEpilogue`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`Max`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`MaxFloat`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ReduceOp`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`Reduction`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SoftMaxBackwardEpilogue`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SoftMaxForwardEpilogue`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SoftMaxForwardWithMulEpilogue`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SumExpFloat`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SumExpfFloat`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)

### Functions

- **`SoftMaxForward_getBlockSize`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SoftMax_getBlockSize`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SpatialSoftMax_getBlockSize`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SpatialSoftMax_getGridSize`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`SpatialSoftMax_getLaunchSizes`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`WriteBpropResults`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`WriteBpropResultsVectorized`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`WriteFpropResults`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`WriteFpropResultsVectorized`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`blockReduce`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`blockReduceWarp`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`blockReduceWarpInverse`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`combine`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`constexpr`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SoftMaxBackward`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SoftMaxBackwardSmem`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SoftMaxForward`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SoftMaxForwardFast`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SoftMaxForwardGmem`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SoftMaxForwardReg`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SoftMaxForwardSmem`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SpatialSoftMaxBackward`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`cunn_SpatialSoftMaxForward`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`dispatch_host_softmax_backward`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`for`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`host_softmax`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`host_softmax_backward`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`if`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ilpReduce`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`is_32bit_representable`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`masked_softmax_backward_cuda`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`masked_softmax_cuda`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`potential_register_count`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`spatialBlockReduceX`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`warp_shfl_down`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)

### Includes

- **`ATen/AccumulateType.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/Dispatch.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/Functions.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/NativeFunctions.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/TensorOperators.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/TensorUtils.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/WrapDimUtils.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/core/Tensor.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/cuda/CUDAContext.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/cuda/NumericLimits.cuh`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/native/IndexingUtils.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/native/cuda/Loops.cuh`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/native/cuda/MemoryAccess.cuh`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/native/cuda/PersistentSoftmax.cuh`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/native/cuda/block_reduce.cuh`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/ops/_log_softmax_backward_data_native.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/ops/_log_softmax_native.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/ops/_masked_softmax_native.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/ops/_softmax_backward_data.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/ops/_softmax_backward_data_native.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/ops/_softmax_native.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`ATen/ops/softmax.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`c10/macros/Macros.h`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)
- **`type_traits`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)

### Namespaces

- **`at`**: [SoftMax.cu_docs.md](./SoftMax.cu_docs.md)


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
