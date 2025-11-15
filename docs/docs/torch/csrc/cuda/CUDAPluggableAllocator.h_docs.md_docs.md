# Documentation: `docs/torch/csrc/cuda/CUDAPluggableAllocator.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/cuda/CUDAPluggableAllocator.h_docs.md`
- **Size**: 9,127 bytes (8.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/cuda/CUDAPluggableAllocator.h`

## File Metadata

- **Path**: `torch/csrc/cuda/CUDAPluggableAllocator.h`
- **Size**: 6,522 bytes (6.37 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <mutex>

namespace torch::cuda::CUDAPluggableAllocator {

#if defined(USE_ROCM)
using streamType = c10::hip::HIPStream;
#else
using streamType = c10::cuda::CUDAStream;
#endif

TORCH_CUDA_CPP_API std::shared_ptr<
    c10::cuda::CUDACachingAllocator::CUDAAllocator>
getCurrentAllocator();
TORCH_CUDA_CPP_API std::shared_ptr<
    c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn);
TORCH_CUDA_CPP_API void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator);

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(
      size_t size,
      c10::DeviceIndex device_idx,
      cudaStream_t stream);
  size_t size;
  c10::DeviceIndex device_idx;
  cudaStream_t stream{};
};

struct TORCH_CUDA_CPP_API CUDAPluggableAllocator
    : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
  CUDAPluggableAllocator(
      std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
      std::function<void(void*, size_t, int, cudaStream_t)> free_fn);

  CUDAPluggableAllocator(CUDAPluggableAllocator& other);
  CUDAPluggableAllocator(CUDAPluggableAllocator&& other) = delete;
  CUDAPluggableAllocator& operator=(const CUDAPluggableAllocator& other) =
      delete;
  CUDAPluggableAllocator& operator=(CUDAPluggableAllocator&& other) = delete;
  ~CUDAPluggableAllocator() override = default;

  void set_init_fn(std::function<void(int)> init_fn);

  void set_reset_fn(std::function<void()> reset_fn);

  void set_memory_fraction_fn(
      std::function<void(double, int)> memory_fraction_fn);

  void set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn);

  void set_record_stream_fn(
      std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn);

  void set_begin_allocate_to_pool(
      std::function<
          void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
          capture_begin_fn);

  void set_end_allocate_to_pool_fn(
      std::function<void(int, c10::cuda::MempoolId_t)> capture_about_to_end_fn);

  void set_release_pool(
      std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn);

  void* malloc(size_t size, c10::DeviceIndex device, cudaStream_t stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  double getMemoryFraction(c10::DeviceIndex device) override;
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override;
  std::vector<c10::cuda::CUDACachingAllocator::StreamSegmentSize>
  getExpandableSegmentSizes(c10::DeviceIndex device) override;
  void emptyCache(c10::cuda::MempoolId_t mempool_id = {0, 0}) override;
  void enable(bool) override {}
  bool isEnabled() const override {
    return true;
  }
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;

  void recordStream(const c10::DataPtr&, streamType stream) override;

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;
  c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot(
      c10::cuda::MempoolId_t mempool) override;
  void beginAllocateToPool(
      c10::DeviceIndex device,
      c10::cuda::MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)>) override;
  void endAllocateToPool(
      c10::DeviceIndex device,
      c10::cuda::MempoolId_t mempool_id) override;
  void releasePool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id)
      override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  c10::cuda::CUDACachingAllocator::ShareableHandle shareIpcHandle(
      void*) override;
  void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when,
      bool clearHistory) override;
  void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  void attachAllocatorTraceTracker(
      c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override;
  std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
  getCheckpointState(c10::DeviceIndex device, at::cuda::MempoolId_t id)
      override;
  c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps)
      override;
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override;
  cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override;
  std::string name() override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;

 protected:
  std::function<void*(size_t, int, cudaStream_t)> alloc_fn_;
  std::function<void(void*, size_t, int, cudaStream_t)> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void()> reset_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn_;
  std::function<
      void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
      begin_allocate_to_pool_fn_;
  std::function<void(int, c10::cuda::MempoolId_t)> end_allocate_to_pool_fn_;
  std::function<void(int, c10::cuda::MempoolId_t)> relase_pool_fn_;
  std::mutex allocator_mutex_;
  // We do the bookkeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

  bool initialized_ = false;
};
} // namespace torch::cuda::CUDAPluggableAllocator

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 38 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `_AllocationMetadata`, `TORCH_CUDA_CPP_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Allocator.h`
- `c10/cuda/CUDAGraphsC10Utils.h`
- `c10/cuda/CUDAMacros.h`
- `c10/cuda/CUDAStream.h`
- `c10/cuda/CUDACachingAllocator.h`
- `mutex`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/cuda`):

- [`python_comm.cpp_docs.md`](./python_comm.cpp_docs.md)
- [`python_comm.h_docs.md`](./python_comm.h_docs.md)
- [`memory_snapshot.cpp_docs.md`](./memory_snapshot.cpp_docs.md)
- [`GdsFile.h_docs.md`](./GdsFile.h_docs.md)
- [`GreenContext.cpp_docs.md`](./GreenContext.cpp_docs.md)
- [`CUDAPluggableAllocator.cpp_docs.md`](./CUDAPluggableAllocator.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`Module.h_docs.md`](./Module.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`nccl.h_docs.md`](./nccl.h_docs.md)


## Cross-References

- **File Documentation**: `CUDAPluggableAllocator.h_docs.md`
- **Keyword Index**: `CUDAPluggableAllocator.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/cuda`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`python_nccl.h_docs.md_docs.md`](./python_nccl.h_docs.md_docs.md)
- [`THCP.h_docs.md_docs.md`](./THCP.h_docs.md_docs.md)
- [`GreenContext.cpp_kw.md_docs.md`](./GreenContext.cpp_kw.md_docs.md)
- [`CUDAPluggableAllocator.cpp_docs.md_docs.md`](./CUDAPluggableAllocator.cpp_docs.md_docs.md)
- [`GdsFile.cpp_kw.md_docs.md`](./GdsFile.cpp_kw.md_docs.md)
- [`python_comm.cpp_kw.md_docs.md`](./python_comm.cpp_kw.md_docs.md)
- [`GdsFile.cpp_docs.md_docs.md`](./GdsFile.cpp_docs.md_docs.md)
- [`Module.cpp_docs.md_docs.md`](./Module.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `CUDAPluggableAllocator.h_docs.md_docs.md`
- **Keyword Index**: `CUDAPluggableAllocator.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
