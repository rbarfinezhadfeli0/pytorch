# Keyword Index: `c10/cuda/CUDACachingAllocator.h`

## File Information

- **Original File**: [c10/cuda/CUDACachingAllocator.h](../../../c10/cuda/CUDACachingAllocator.h)
- **Documentation**: [`CUDACachingAllocator.h_docs.md`](./CUDACachingAllocator.h_docs.md)
- **Folder**: `c10/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AllocatorConfigInfo`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`AllocatorState`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`AnnotationEntry`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`BlockInfo`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`C10_CUDA_API`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`CUDAAllocator`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`CheckpointDelta`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`RecordContext`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`SegmentInfo`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`ShareableHandle`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`SnapshotInfo`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`StreamSegmentSize`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`TraceEntry`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`which`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)

### Functions

- **`attachAllocatorTraceTracker`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`attachOutOfMemoryObserver`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`cacheInfo`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`checkPoolLiveAllocations`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`createOrIncrefPool`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`emptyCache`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`enable`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`enablePeerAccess`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`endAllocateToPool`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`getDeviceStats`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`getMemoryFraction`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`getPoolUseCount`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`getUserMetadata`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`init`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`isEnabled`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`isHistoryEnabled`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`memcpyAsync`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`name`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`popCompileContext`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`pushCompileContext`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`raw_delete`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`recordAnnotation`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`recordHistory`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`recordStream`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`recordUserMetadata`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`releasePool`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`resetAccumulatedStats`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`resetPeakStats`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`setCheckpointPoolState`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`setMemoryFraction`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`setUseOnOOM`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`setUserMetadata`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`shareIpcHandle`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`snapshot`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)

### Includes

- **`atomic`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/core/AllocatorConfig.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/core/CachingDeviceAllocator.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/cuda/CUDAAllocatorConfig.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/cuda/CUDAGraphsC10Utils.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/cuda/CUDAMacros.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/cuda/CUDAStream.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/util/ApproximateClock.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/util/Exception.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`c10/util/Registry.h`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`cstddef`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`cstdint`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`functional`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`memory`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`string`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`unordered_set`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`utility`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)

### Namespaces

- **`c10`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)
- **`rather`**: [CUDACachingAllocator.h_docs.md](./CUDACachingAllocator.h_docs.md)


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
