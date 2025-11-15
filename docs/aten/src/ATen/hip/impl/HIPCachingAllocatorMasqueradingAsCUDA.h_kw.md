# Keyword Index: `aten/src/ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h`

## File Information

- **Original File**: [aten/src/ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h](../../../../../../aten/src/ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h)
- **Documentation**: [`HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md`](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **Folder**: `aten/src/ATen/hip/impl`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`DataPtr`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)

### Functions

- **`attachAllocatorTraceTracker`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`attachOutOfMemoryObserver`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`cacheInfo`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`checkPoolLiveAllocations`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`createOrIncrefPool`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`emptyCache`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`enable`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`enablePeerAccess`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`endAllocateToPool`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`getDeviceStats`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`getMemoryFraction`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`getPoolUseCount`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`init`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`isEnabled`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`isHistoryEnabled`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`memcpyAsync`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`name`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`popCompileContext`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`pushCompileContext`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`raw_delete`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`recordAnnotation`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`recordHistory`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`releasePool`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`resetAccumulatedStats`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`resetPeakStats`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`setCheckpointPoolState`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`setMemoryFraction`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`setUseOnOOM`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`shareIpcHandle`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`snapshot`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)

### Includes

- **`ATen/hip/impl/HIPAllocatorMasqueradingAsCUDA.h`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`c10/hip/HIPCachingAllocator.h`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)

### Namespaces

- **`HIPCachingAllocatorMasqueradingAsCUDA`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`c10`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)
- **`hip`**: [HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md](./HIPCachingAllocatorMasqueradingAsCUDA.h_docs.md)


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
