# Keyword Index: `c10/xpu/XPUCachingAllocator.cpp`

## File Information

- **Original File**: [c10/xpu/XPUCachingAllocator.cpp](../../../c10/xpu/XPUCachingAllocator.cpp)
- **Documentation**: [`XPUCachingAllocator.cpp_docs.md`](./XPUCachingAllocator.cpp_docs.md)
- **Folder**: `c10/xpu`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AllocParams`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`Block`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`BlockPool`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`DeviceCachingAllocator`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`ExpandableSegment`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`SegmentRange`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`XPUAllocator`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`the`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)

### Functions

- **`BlockComparatorAddress`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`BlockComparatorSize`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`add_allocated_block`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`alloc_block`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`assertValidDevice`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`device`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`emptyCache`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`enablePeerAccess`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`free`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`free_block`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`getDeviceStats`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`getMemoryFraction`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`getStats`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`get_allocation_size`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`get_free_block`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`get_stat_types_for_pool`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`if`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`init`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`insert_events`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`is_split`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`local_raw_delete`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`malloc`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`map`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`map_block`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`numSegments`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`process_events`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`rangeFromHandles`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`raw_delete`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`recordStream`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`release_block`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`release_blocks`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`release_cached_blocks`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`release_expandable_segment`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`resetAccumulatedStats`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`resetPeakStats`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`round_size`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`segmentLeft`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`segmentRight`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`setMemoryFraction`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`should_split`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`size`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`splice`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`synchronize_and_free_events`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`trimHandles`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`try_merge_blocks`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`unmap`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`unmapHandles`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`unmap_block`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`while`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)

### Includes

- **`c10/util/flat_hash_map.h`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`c10/util/irange.h`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`c10/xpu/XPUCachingAllocator.h`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`deque`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`mutex`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`set`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`vector`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)

### Namespaces

- **`c10`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`class`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)
- **`sycl`**: [XPUCachingAllocator.cpp_docs.md](./XPUCachingAllocator.cpp_docs.md)


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
