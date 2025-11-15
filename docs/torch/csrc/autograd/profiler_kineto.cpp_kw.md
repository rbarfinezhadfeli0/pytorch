# Keyword Index: `torch/csrc/autograd/profiler_kineto.cpp`

## File Information

- **Original File**: [torch/csrc/autograd/profiler_kineto.cpp](../../../../torch/csrc/autograd/profiler_kineto.cpp)
- **Documentation**: [`profiler_kineto.cpp_docs.md`](./profiler_kineto.cpp_docs.md)
- **Folder**: `torch/csrc/autograd`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AddGenericMetadata`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`AddTensorboardFields`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`KinetoThreadLocalState`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`MetadataBase`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`OpArgData`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`ProfilerStateInfo`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)

### Functions

- **`_reportVulkanEventToProfiler`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`acc_get_device_type`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`addMetadata`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`constexpr`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`disableProfilerInChildThread`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`enableProfiler`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`enableProfilerInChildThread`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`enableProfilerWithEventPostProcess`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`exportMemoryProfile`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`for`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`getTimeNs`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`hasKinetoActivity`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`if`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`invokeCallback`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`isProfilerEnabledInMainThread`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`materializeOpEvents`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`onFunctionExit`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`parseArgData`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`pausePython`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`prepareProfiler`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`pushProfilingCallbacks`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`reportBackendEventToActiveKinetoProfiler`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`reportVulkanEventToProfiler`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`resumePython`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`setEventPostProcessingCallback`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`startMemoryProfile`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`stopMemoryProfile`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`toggleCPUCollectionDynamic`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`toggleCollectionDynamic`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`togglePythonCollectionDynamic`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`toggleTorchOpCollectionDynamic`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)

### Includes

- **`ApproximateClock.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`c10/macros/Export.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`c10/util/ApproximateClock.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`c10/util/Exception.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`c10/util/flat_hash_map.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`c10/util/irange.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`c10/util/overloaded.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`cstring`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`libkineto.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`stdexcept`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`time_since_epoch.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/autograd/profiler_kineto.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/api.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/collection.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/containers.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/events.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/kineto_shim.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/orchestration/observer.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/perf.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/standalone/itt_observer.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/standalone/nvtx_observer.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/standalone/privateuse1_observer.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch/csrc/profiler/util.h`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`utility`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)

### Namespaces

- **`autograd`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`profiler`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`torch`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`tracer`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)
- **`void`**: [profiler_kineto.cpp_docs.md](./profiler_kineto.cpp_docs.md)


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
