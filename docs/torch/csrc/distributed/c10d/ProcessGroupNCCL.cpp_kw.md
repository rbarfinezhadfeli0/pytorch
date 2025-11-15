# Keyword Index: `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`

## File Information

- **Original File**: [torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp](../../../../../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)
- **Documentation**: [`ProcessGroupNCCL.cpp_docs.md`](./ProcessGroupNCCL.cpp_docs.md)
- **Folder**: `torch/csrc/distributed/c10d`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`var`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)

### Functions

- **`_ncclMemFree`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`attachAllocatorHooks`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`cacheAllocatorDeregisterHook`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`cacheAllocatorRegisterHook`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`check_gpu_single_tensor`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`check_gpu_tensors_same_device`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`check_same_size`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`collective`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`collectiveCoalesced`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`computeDeltaMS`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`dump_nccl_trace`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`dump_nccl_trace_json`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`errorIfCapturingNonCapturableNCCL`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`for`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`getDevice`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`getExceptionMsgFromExceptionPtr`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`getKeyFromDevice`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`getKeySendRecv`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`getNcclAbortedCommStoreKey`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`getNcclReduceOp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`getRootIndex`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`if`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`isUnsupportedFloat8`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`pointToPoint`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`reset_nccl_trace`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`routine`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`shouldAllCommunicatorsRegisterAllTensors`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`switch`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`syncStream`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`unpackPreMulSum`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)

### Includes

- **`ATen/cuda/CUDAContext.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/core/DeviceType.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/cuda/CUDAAllocatorConfig.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/cuda/CUDAGraphsC10Utils.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/cuda/CUDAGuard.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/util/Exception.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/util/Logging.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/util/WaitCounter.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/util/hash.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/util/irange.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10/util/thread_name.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`exception`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`map`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`mutex`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`nlohmann/json.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`optional`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`sstream`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`stdexcept`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/cuda/CUDAPluggableAllocator.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/cuda/nccl.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/FlightRecorder.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/NCCLUtils.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/NanCheck.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/ParamCommsUtils.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/PrefixStore.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/TraceUtils.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/Utils.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/csrc/distributed/c10d/cuda/utils.hpp`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`torch/torch.h`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`tuple`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`utility`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)

### Namespaces

- **`c10`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`c10d`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)
- **`intra_node_comm`**: [ProcessGroupNCCL.cpp_docs.md](./ProcessGroupNCCL.cpp_docs.md)


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
