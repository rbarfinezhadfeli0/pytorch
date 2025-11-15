# Documentation: `c10/cuda/CUDACachingAllocator.cpp`

## File Metadata

- **Path**: `c10/cuda/CUDACachingAllocator.cpp`
- **Size**: 162,911 bytes (159.09 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Gauge.h>
#include <c10/util/Logging.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/env.h>
#include <c10/util/error.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/static_tracepoint.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <regex>
#include <set>
#include <stack>
#include <thread>
#include <utility>
#include <vector>

TORCH_SDT_DEFINE_SEMAPHORE(malloc)
TORCH_SDT_DEFINE_SEMAPHORE(free)

// add these definitions so that we can compile with CUDA < 12.3
// borrowed from
// https://github.com/NVIDIA/nccl/blob/3ea7eedf3b9b94f1d9f99f4e55536dfcbd23c1ca/src/include/p2p.h#L20
#if CUDA_VERSION < 12030
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#define CU_IPC_HANDLE_SIZE 64
typedef struct CUmemFabricHandle_st {
  unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

namespace c10 {

// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback)

namespace cuda::CUDACachingAllocator {

using namespace c10::CachingAllocator;
using namespace c10::CachingDeviceAllocator;

namespace Native {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= max_split_size are not allowed
//   to be split. These oversize cached blocks will still satisfy requests
//   within 1MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

/**
 * Note [Interaction with CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Graph capture performs a dry run of a region of execution, freezing all CUDA
 * work (and virtual addresses used during that work) into a "graph." The graph
 * may be "replayed" like a single giant kernel, with greatly reduced CPU
 * overhead as well as modestly improved GPU performance.
 *
 * Because capture bakes in memory addresses, the memory used during capture
 * must be available for the graph to use during replay. DeviceCachingAllocator
 * assigns and frees memory eagerly and dynamically, so if we're not careful
 * about managing graphs' memory, at replay time those memory addresses could be
 * used by other tensors.
 *
 * To guarantee a graph's baked in addresses are safe to reuse in replay,
 * DeviceAllocator satisfies allocations from a graph-private memory pool during
 * capture, and doesn't begin cudaFreeing those addresses until the graph is
 * destroyed.
 *
 * Within the private pool, allocations are freed and reassigned as usual during
 * capture. Memory regions will be used in a consistent order during replay. So
 * a private pool doesn't use memory more wastefully than the default pools
 * during capture, but it does reserve its high-water mark of used memory away
 * from the default pools as long as the capture(s) it served survive
 * (regardless whether those captures are idle or replaying).
 *
 * CUDAGraph's requests for private pools are mediated by
 * DeviceAllocator::notifyCaptureBegin,
 *                  notifyCaptureAboutToEnd,
 *                  notifyCaptureEnded,
 *                  notifyCaptureDestroy.
 */

static char SHAREABLE_HANDLE_VERSION = 2;
enum ShareableHandleType : char {
  SHAREABLE_CUDA_MALLOC = 'c',
  SHAREABLE_CUDA_EXPANDABLE_SEGMENT = 'e'
};

namespace {

using stream_set = ska::flat_hash_set<cuda::CUDAStream>;

void decrease_stat_array(
    StatArray& stat_array,
    size_t amount,
    const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        stat_array[stat_type].decrease(amount);
      });
}

struct Block;
struct PrivatePool;
typedef bool (*Comparison)(const Block*, const Block*);
static bool BlockComparatorSize(const Block* a, const Block* b);
static bool BlockComparatorAddress(const Block* a, const Block* b);

struct BlockPool {
  BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  // Do not insert a Block to blocks directly; use insert_into_blocks(),
  // instead.
  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool is_small;
  PrivatePool* owner_PrivatePool;
  int64_t get_free_blocks_call_count{0};

  // Add a Block into blocks set with updating gc counter.
  std::pair<std::set<Block*, Comparison>::iterator, bool> insert_into_blocks(
      Block* block);

  MempoolId_t owner_MempoolId() const;
};

struct ExpandableSegment;

struct Block {
  c10::DeviceIndex device; // gpu
  cudaStream_t stream; // allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  bool mapped{true}; // is the virtual address range this Block references
                     // backed by physical pages. Always true when
                     // expandable_segment_ is null. When false
                     // This Block will be aligned to the segment size
                     // of its expandable_segment_.
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding CUDA events
  int64_t gc_count_base{0}; // get_free_blocks_call_count when Block is inserted
  std::shared_ptr<GatheredContext> context_when_allocated;
  // only set for the first block in the segment (when prev == null)
  // this records the frame information when cudaMalloc was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<GatheredContext> context_when_segment_allocated;

  ExpandableSegment* expandable_segment_{nullptr};

  Block(
      c10::DeviceIndex device,
      cudaStream_t stream,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(c10::DeviceIndex device, cudaStream_t stream, size_t size)
      : device(device), stream(stream), size(size), requested_size(0) {}

  size_t gc_count() {
    TORCH_INTERNAL_ASSERT(pool);
    return static_cast<int>(pool->get_free_blocks_call_count - gc_count_base);
  }

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
  void splice(Block* before, Block* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }
};

std::pair<std::set<Block*, Comparison>::iterator, bool> BlockPool::
    insert_into_blocks(Block* block) {
  block->gc_count_base = get_free_blocks_call_count;
  return blocks.insert(block);
}

struct SegmentRange {
  char* ptr;
  size_t size;
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)

/*
Note [Expandable Segments]

Rationale

For large (>2MB) allocations, the allocator calls cudaMalloc to get allocations
that are the same size as what the user requests. In the future, parts of these
allocations can be reused for other requests if they are free. This works well
when the program makes many requests of exactly the same size or of sizes that
even multiples of that size. Many deep learning models follow this behavior.
However, one common exception is when the batch size changes slightly from one
iteration to the next, e.g. in batched inference. When the program runs
initially with batch size N, it will make allocations appropriate for that size.
If in the future, it runs at size N - 1, the existing allocations will still be
big enough. However, if it runs at size N + 1, then it will have to make new
allocations that are slightly larger. Not all the tensors are the same size.
Some might be (N + 1)*A and others (N + 1)*A*B where A and B are some non-batch
dimensions in the model. Because the allocator reuses existing allocations when
they are big enough, some number of (N + 1)*A allocations will actually fit in
the already existing N*B*A segments, though not perfectly. As the model runs it
will partially fill up all of these segments leaving unusable free slices of
memory at the end of these segments. The allocator at some point will need to
cudaMalloc a new (N + 1)*A*B segment. If there is not enough memory, there is
now no way to recover the slices of memory that are free at the end of existing
segments. With models 50+ layers deep, this pattern might repeat 50+ times
creating many slivers.

Approach

Expandable segments allows the allocator to create a segment initially and then
expand its size later when more memory is needed. Instead of making one segment
per allocation, it tries to make one segment (per stream) that grows as
necessary. Now when the N + 1 case runs, the allocations will tile nicely into
the one large segment until it fills up. Then more memory is requested and
appended to the end of the segment. This process does not create as many slivers
of unusable memory, so it is more likely to succeed at finding this memory.

Implementation

The expandable_segments:True option is used to enable/disable this behavior. We
use cuda's low-level memory APIs, which are similar to mmap, to extend the
memory segments. These APIs separate the allocation of physical memory
(cuMemCreate) from the allocation of virtual address space (cuMemAddressReserve)
and the associate between them cuMemMap/cuMemSetAccess.

When we allocate a new segment, we allocate enough address space to map
basically the entire physical memory of the GPU (there is 256TiB of address
space), but we only map enough physical memory to handle the current amount of
memory needed by the program. As more is requested, we add more physical memory
to the segment. This can work at the granularity of GPU pages which are 2MiB
currently.

If we end up out of memory, we can unmap all the memory in our segment
corresponding to empty physical pages, and return it to CUDA for use at another
address in the segment or in a segment for a different stream.

A current limitation of CUDA's API is that physical memory
(CUmemGenericAllocationHandle) cannot be split up after it is mapped even if the
handle holds multiple GPU pages. The cost to map/unmap memory is proportional to
the number of physical memory chunks that were allocated (mapping 10 separately
allocated 2MiB pages takes 10x time compared to mapping one 20MiB physical
allocation of 10 pages).  Changing memory mappings also appears to involve at
least some synchronous actions with the GPU and so should be considered an
expensive operation. To limit overhead, we use 2MiB pages for our small pool and
20MiB pages for our large pool. Initially allocation using expandable_blocks
will be slower than cudaMalloc, though still in the milliseconds range for
mapping the entire memory.

When mapping new memory to expand the segment, we look for the lowest address at
which we can fit a new allocation by adding new pages. Normally this will be at
the end of the block. But if have previously unmapped blocks earlier in the
segment during an OOM, it will first try to fill in those gaps to keep the
segment as a single block. By allocating at the lowest address we encourage
the split up parts of the block to merge into a single block again, reducing
fragmentation potential.

Allocation of blocks in the segment uses the same best-fit heuristics of the
rest of the allocator.

Expandable blocks can be enabled/disabled throughout the run of a program. When
disabled, the allocator will not put new allocations in an expandable block.

Limitations

* Slightly slower initial memory allocation speed.
* IPC of cuda tensors (e.g. for multiprocess dataloaders) is not supported.
However, it is possible to temporarily disable (expandable_segments:False) the
bevhavior for allocator tensors that need to be used cross-process.
* CUDA runtime APIs related to sharing memory across process
(cudaDeviceEnablePeerAccess) do not work for memory allocated with cuMemMap.
Instead these mapping have to be done manually. The allocator now has an
`enablePeerAccess` method to do this.
*/

struct ExpandableSegment {
  ExpandableSegment(
      c10::DeviceIndex device,
      std::optional<cudaStream_t> stream,
      size_t segment_size,
      std::vector<c10::DeviceIndex> peers)
      : device_(device),
        stream_(stream),
        // 2MB for small pool, 20MB for large pool
        segment_size_(segment_size),
        peers_(std::move(peers)) {
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
    mapped_size_ = 0;
    // we allocate enough address space for 1 1/8 the total memory on the GPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    max_handles_ = numSegments(prop.totalGlobalMem + prop.totalGlobalMem / 8);
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressReserve_(
        &ptr_, segment_size_ * max_handles_, 0ULL, 0, 0ULL));
  }
  ExpandableSegment(const ExpandableSegment&) = delete;
  ExpandableSegment(ExpandableSegment&&) = delete;
  ExpandableSegment operator=(const ExpandableSegment&) = delete;
  ExpandableSegment operator=(ExpandableSegment&&) = delete;

  // begin must be aligned to segment_size_.
  // returns the actual range mapped, which may be
  // greater than requested if size is not aligned to segment_size_.
  // return size of 0 indicates OOM
  // return nullptr indicates the handle type is not supported.
  SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }

    // if the handle type is not specified, try to use fabric handle first.
    // if it fails, use posix file handle
    if (CUDAAllocatorConfig::expandable_segments_handle_type() ==
        Expandable_Segments_Handle_Type::UNSPECIFIED) {
      CUDAAllocatorConfig::set_expandable_segments_handle_type(
          Expandable_Segments_Handle_Type::FABRIC_HANDLE);
      auto output = map(range);
      if (output.ptr != nullptr) {
        return output;
      }
      // if fabric handle is not supported, use posix file handle.
      CUDAAllocatorConfig::set_expandable_segments_handle_type(
          Expandable_Segments_Handle_Type::POSIX_FD);
      return map(range);
    }

    while (end > handles_.size()) {
      handles_.emplace_back(std::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i));
      CUmemGenericAllocationHandle handle = 0;
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
#ifndef FBCODE_CAFFE2
      if (CUDAAllocatorConfig::expandable_segments_handle_type() !=
          Expandable_Segments_Handle_Type::FABRIC_HANDLE) {
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      } else {
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
      }
#endif
      int flag = 0;
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuDeviceGetAttribute_(
          &flag,
          CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
          device_));
      if (flag)
        prop.allocFlags.gpuDirectRDMACapable = 1;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      // NOLINTNEXTLINE(bugprone-signed-char-misuse)
      prop.location.id = static_cast<int>(device_);
      auto status =
          DriverAPI::get()->cuMemCreate_(&handle, segment_size_, &prop, 0);
      if (status != CUDA_SUCCESS) {
        if (status == CUDA_ERROR_OUT_OF_MEMORY) {
          for (auto j : c10::irange(begin, i)) {
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            auto h = handles_.at(j).value();
            handles_.at(j) = std::nullopt;
            C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h.handle));
          }
          trimHandles();
          return rangeFromHandles(begin, begin);
        } else if (
            CUDAAllocatorConfig::expandable_segments_handle_type() ==
            Expandable_Segments_Handle_Type::FABRIC_HANDLE) {
          // we are testing if we can use fabric handle.
          // if we can, we will use it.
          // if we can't, we will use posix file handle.
          // so we should not return an error here.
          // in practice, we can get CUDA_ERROR_NOT_SUPPORTED or
          // CUDA_ERROR_NOT_PERMITTED to be safe, any non out-of-memory error is
          // considered as the handle type is not supported. if the handle type
          // is not supported, return a null range to indicate it.
          return SegmentRange(nullptr, 0);
        } else {
          C10_CUDA_DRIVER_CHECK(status);
        }
      }
      handles_.at(i) = Handle{handle, std::nullopt};
    }
    mapAndSetAccess(begin, end);
    return rangeFromHandles(begin, end);
  }

  // unmaps all the completely empty segment_size_ segments between
  // [begin, begin + size), returns the offset where the range begin,
  // and the actual size unmapped (multiple of segment_size_)
  SegmentRange unmap(SegmentRange range) {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    mapped_size_ -= (end - begin) * segment_size_;
    return rangeFromHandles(begin, end);
  }

  // Setup IPC sharing for range.
  // Returns the (larger) range that was actually shared.
  // Serializes data to std::ostream that can be passed to the
  // other process, and then restored as an exapandable segment
  // via ExpandableSegment::fromShared(istream);
  SegmentRange share(SegmentRange range, std::ostream& buf) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);

    // header.pid needs to be padded with 4 bytes and initialized with
    // 0 values ​​to avoid random padding of different bytes each time,
    // thereby ensuring that the handle can be correctly matched in
    // ipcMemHandle_to_devptr.
    ShareHeader header{};
    header.pid = getpid();
    header.segment_size = segment_size_;
    header.num_handles = end - begin;

    buf.write(reinterpret_cast<const char*>(&header), sizeof(ShareHeader));
    for (auto i : c10::irange(begin, end)) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      auto& handle = handles_.at(i).value();
      if (CUDAAllocatorConfig::expandable_segments_handle_type() !=
          Expandable_Segments_Handle_Type::FABRIC_HANDLE) {
        if (!handle.shareable_handle) {
          int fd = 0;
          C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemExportToShareableHandle_(
              &fd, handle.handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
          handle.shareable_handle = fd;
          LOG(INFO) << "use posix fd to share expandable segments.";
        }
        TORCH_CHECK(
            handle.shareable_handle != std::nullopt,
            "shareable_handle is null");
        buf.write(
            reinterpret_cast<const char*>(&*handle.shareable_handle),
            sizeof(int));
      } else {
        if (!handle.shareable_handle) {
          CUmemFabricHandle fabric_handle;
          C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemExportToShareableHandle_(
              &fabric_handle, handle.handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
          handle.shareable_handle = fabric_handle;
          LOG(INFO) << "use fabric handle to share expandable segments.";
        }
        TORCH_CHECK(
            handle.shareable_handle != std::nullopt,
            "shareable_handle is null");
        buf.write(
            reinterpret_cast<const char*>(&*handle.shareable_handle),
            sizeof(CUmemFabricHandle));
      }
    }
    return rangeFromHandles(begin, end);
  }

  static std::unique_ptr<ExpandableSegment> fromShared(
      c10::DeviceIndex device,
      std::vector<c10::DeviceIndex> peers,
      std::istream& buf) {
    ShareHeader header{};
    buf.read(reinterpret_cast<char*>(&header), sizeof(ShareHeader));
    auto segment = std::make_unique<ExpandableSegment>(
        device, std::nullopt, header.segment_size, std::move(peers));
// older build setups (e.g. multiwheels) do not have this syscall, added 2020
// but the kernel on the system might still support it.
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif
    if (CUDAAllocatorConfig::expandable_segments_handle_type() !=
        Expandable_Segments_Handle_Type::FABRIC_HANDLE) {
      auto pidfd = syscall(SYS_pidfd_open, header.pid, 0);
      TORCH_CHECK(
          pidfd != -1 || errno != ENOSYS,
          "The kernel on this machine does not support the pidfd_open syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. "
          "Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.");
      TORCH_CHECK(pidfd != -1, "pidfd_open:", c10::utils::str_error(errno));
      for (auto i : c10::irange(header.num_handles)) {
        (void)i;
        int fd = 0;
        buf.read(reinterpret_cast<char*>(&fd), sizeof(int));
        auto myfd = syscall(SYS_pidfd_getfd, pidfd, fd, 0);
        if (myfd == -1) {
          auto err = errno;
          close(static_cast<int>(pidfd));
          for (auto& h : segment->handles_) {
            C10_CUDA_DRIVER_CHECK(
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                DriverAPI::get()->cuMemRelease_(h.value().handle));
            h = std::nullopt;
          }
          TORCH_CHECK(
              err != ENOSYS,
              "The kernel on this machine does not support the pidfd_getfd syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. "
              "Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.");
          TORCH_CHECK(false, "pidfd_getfd: ", c10::utils::str_error(err));
        }
        CUmemGenericAllocationHandle handle = 0;
        C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemImportFromShareableHandle_(
            &handle,
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            (void*)(uintptr_t)myfd,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        LOG(INFO) << "use posix fd to import expandable segments.";
        close(static_cast<int>(myfd));
        segment->handles_.emplace_back(Handle{handle, std::nullopt});
      }
      close(static_cast<int>(pidfd));
    } else {
      for (auto i : c10::irange(header.num_handles)) {
        (void)i;
        CUmemFabricHandle fabric_handle;
        buf.read(
            reinterpret_cast<char*>(&fabric_handle), sizeof(CUmemFabricHandle));
        CUmemGenericAllocationHandle handle = 0;
        C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemImportFromShareableHandle_(
            &handle,
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            (void*)&fabric_handle,
            CU_MEM_HANDLE_TYPE_FABRIC));
        LOG(INFO) << "use fabric handle to import expandable segments.";
        segment->handles_.emplace_back(Handle{handle, std::nullopt});
      }
    }
    segment->mapAndSetAccess(0, header.num_handles);
    return segment;
  }

  char* ptr() const {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<char*>(ptr_);
  }

  size_t size() const {
    return max_handles_ * segment_size_;
  }

  cudaStream_t getStream() {
    return *stream_;
  }

  size_t getMappedSize() const {
    return mapped_size_;
  }

  size_t getSegmentSize() const {
    return segment_size_;
  }

  void addPeer(c10::DeviceIndex device) {
    peers_.push_back(device);
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { setAccess(device, begin, end); });
  }

  ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressFree_(
        ptr_, segment_size_ * max_handles_));
  }

 private:
  void setAccess(c10::DeviceIndex device, size_t begin, size_t end) {
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    desc.location.id = static_cast<int>(device);
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemSetAccess_(
        ptr_ + begin * segment_size_, (end - begin) * segment_size_, &desc, 1));
  }

  void mapAndSetAccess(size_t begin, size_t end) {
    for (auto i : c10::irange(begin, end)) {
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemMap_(
          ptr_ + i * segment_size_,
          segment_size_,
          0,
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          handles_.at(i).value().handle,
          0ULL));
    }
    mapped_size_ += (end - begin) * segment_size_;
    setAccess(device_, begin, end);
    for (auto p : peers_) {
      setAccess(p, begin, end);
    }
  }

  void unmapHandles(size_t begin, size_t end) {
    // note: unlike cudaFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::cuda::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    if (stream_) {
      C10_CUDA_CHECK(cudaStreamSynchronize(*stream_));
    } else {
      cuda::CUDAGuard device_guard(device_);
      C10_CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (auto i : c10::irange(begin, end)) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      Handle h = handles_.at(i).value();
      handles_.at(i) = std::nullopt;
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemUnmap_(
          ptr_ + segment_size_ * i, segment_size_));
      if (h.shareable_handle) {
        close(std::get<int>(*h.shareable_handle));
      }
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h.handle));
    }
    trimHandles();
  }
  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }
  void forEachAllocatedRange(const std::function<void(size_t, size_t)>& fn) {
    size_t start = 0;
    for (auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }
  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }
  size_t segmentLeft(char* p) {
    auto size = p - ptr();
    return size / segment_size_;
  }
  size_t segmentRight(char* p) {
    auto size = p - ptr();
    return numSegments(size);
  }
  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }
  c10::DeviceIndex device_;
  std::optional<cudaStream_t> stream_;
  CUdeviceptr ptr_{};
  size_t segment_size_;
  size_t mapped_size_;
  size_t max_handles_;
  struct Handle {
    CUmemGenericAllocationHandle handle;
    std::optional<std::variant<int, CUmemFabricHandle>> shareable_handle;
  };
  struct ShareHeader {
    pid_t pid;
    size_t segment_size;
    size_t num_handles;
  };
  std::vector<std::optional<Handle>> handles_;
  // devices on which this memory should be mapped in addition
  // to the device where the physical memory lives (device_).
  std::vector<c10::DeviceIndex> peers_;
};
#else
struct ExpandableSegment {
  ExpandableSegment(
      c10::DeviceIndex device,
      std::optional<cudaStream_t> stream,
      size_t segment_size,
      std::vector<c10::DeviceIndex> peers) {
    TORCH_INTERNAL_ASSERT(false, "expandable segment not supported");
  }
  SegmentRange map(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  SegmentRange unmap(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  SegmentRange share(SegmentRange range, std::ostream& ss) {
    return SegmentRange(nullptr, 0);
  }
  static std::unique_ptr<ExpandableSegment> fromShared(
      c10::DeviceIndex device,
      std::vector<c10::DeviceIndex> peers,
      std::istream& buf) {
    return {};
  }
  char* ptr() const {
    return nullptr;
  }
  size_t size() const {
    return 0;
  }
  cudaStream_t getStream() {
    return nullptr;
  }

  size_t getMappedSize() const {
    return 0;
  }

  size_t getSegmentSize() const {
    return 0;
  }
  void addPeer(c10::DeviceIndex device) {}
};
#endif

// BlockState, BlockPoolState, and PrivatePoolState contain the information
// needed to reconstruct a private pool to a previous state. See note
// [Checkpointing PrivatePoolState]
struct BlockState {
  c10::DeviceIndex device = 0;
  cudaStream_t stream = nullptr;
  stream_set stream_uses;
  size_t size = 0;
  void* ptr = nullptr;
  bool allocated = false;
  int64_t gc_count_base = 0;
  // maintain invariant that event_count == 0 ;
  // history will be left alone in checkpoint

  explicit BlockState(Block* block);
};

struct SegmentState {
  std::vector<BlockState> blocks;
  bool is_small = false;

  explicit SegmentState(Block* head);
};

struct PrivatePoolState : AllocatorState {
  // omitting use_count, and cudaMalloc_count as they remain the same
  MempoolId_t owner_id = {0, 0};

  std::vector<SegmentState> segments;

  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Block*>& private_pool_head_blocks);
};

struct RestoreResult {
  std::vector<void*> allocations_freed;
  std::vector<Block*> allocations_created;
};

bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}
bool BlockComparatorAddress(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams {
  AllocParams(
      c10::DeviceIndex device,
      size_t size,
      cudaStream_t stream,
      BlockPool* pool,
      size_t alloc_size)
      : search_key(device, stream, size), pool(pool), alloc_size(alloc_size) {}

  c10::DeviceIndex device() const {
    return search_key.device;
  }
  cudaStream_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block{nullptr};
  StatTypes stat_types = {false};
  cudaError_t err{cudaSuccess};
};

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>>;
  // TODO: Explicit device count
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(c10::DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](cudaEvent_t* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<cudaEvent_t>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    auto new_ptr = std::make_unique<cudaEvent_t>();
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(new_ptr.get(), cudaEventDisableTiming));

    return Event(new_ptr.release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(hardware_destructive_interference_size) std::mutex mutex_;
    std::vector<std::unique_ptr<cudaEvent_t>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// CUDA graphs helper
struct PrivatePool {
  explicit PrivatePool(MempoolId_t id, CUDAAllocator* allocator = nullptr)
      : id(std::move(id)),
        allocator_(allocator),
        large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  PrivatePool& operator=(PrivatePool&&) = delete;
  ~PrivatePool() = default;

  MempoolId_t id{0, 0};
  // Number of live graphs using this pool
  int use_count{1};
  // Number of unfreed cudaMallocs made for this pool. When use_count and
  // cudaMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int cudaMalloc_count{0};
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critical though,
  // I'd rather not add more logic to it.
  CUDAAllocator* allocator_;
  BlockPool large_blocks;
  BlockPool small_blocks;

 public:
  CUDAAllocator* allocator() {
    return allocator_;
  }
};

MempoolId_t BlockPool::owner_MempoolId() const {
  if (owner_PrivatePool) {
    return owner_PrivatePool->id;
  } else {
    return {0, 0};
  }
}

BlockState::BlockState(Block* block)
    : device(block->device),
      stream(block->stream),
      stream_uses(block->stream_uses),
      size(block->size),
      ptr(block->ptr),
      allocated(block->allocated),
      gc_count_base(block->gc_count_base) {
  TORCH_CHECK(
      block->event_count == 0,
      "Events should have synchronized when checkpointing block");
}

SegmentState::SegmentState(Block* head) {
  TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
  is_small = head->pool->is_small;

  for (Block* curr = head; curr != nullptr; curr = curr->next) {
    blocks.emplace_back(curr);
  }
}

PrivatePoolState::PrivatePoolState(
    MempoolId_t pool_id,
    const std::vector<Block*>& private_pool_head_blocks)
    : owner_id(std::move(pool_id)) {
  for (Block* head : private_pool_head_blocks) {
    segments.emplace_back(head);
  }
}

cudaError_t allocPrimitive(void** ptr, size_t size, AllocParams& p) {
  if (p.pool->owner_PrivatePool && p.pool->owner_PrivatePool->allocator()) {
    *ptr = p.pool->owner_PrivatePool->allocator()->raw_alloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
  } else {
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(ptr, size));
  }
}

cudaError_t cudaMallocMaybeCapturing(void** ptr, size_t size, AllocParams& p) {
  if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
      at::cuda::CaptureStatus::None) {
    return allocPrimitive(ptr, size, p);
  } else {
    // It's ok to capture cudaMallocs, as long as we never cudaFree those
    // addresses before replay.
    // Capturing cudaMalloc behaves nicely: it gives the graph new VA,
    // but is ignored (won't leakily allocate new memory) in replays.
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    return allocPrimitive(ptr, size, p);
  }
}

template <class T>
class RingBuffer {
 public:
  RingBuffer() {
    // alloc_trace is a pointer because we need to intentionally
    // leak this on deallocation it can hold references to Python
    // state which will already be destroyed when we are in exit handlers
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    alloc_trace = new std::vector<T>();
  }

  void setMaxEntries(size_t size) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    alloc_trace_max_entries_ = std::max(static_cast<size_t>(1), size);
  }

  void insertEntries(const T& entry) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    if (alloc_trace->size() < alloc_trace_max_entries_) {
      alloc_trace->emplace_back(entry);
    } else {
      (*alloc_trace)[alloc_trace_next++] = entry;
      if (alloc_trace_next == alloc_trace_max_entries_) {
        alloc_trace_next = 0;
      }
    }
  }

  void getEntries(std::vector<T>& result) const {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    result.reserve(result.size() + alloc_trace->size());
    std::rotate_copy(
        alloc_trace->begin(),
        std::next(alloc_trace->begin(), alloc_trace_next),
        alloc_trace->end(),
        std::back_inserter(result));
  }

  void clear() {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    alloc_trace_next = 0;
    alloc_trace->clear();
  }

 private:
  size_t alloc_trace_max_entries_ = 1;

  // Both alloc_trace and alloc_trace_next needs to be used
  // under alloc_trace_lock.
  mutable std::mutex alloc_trace_lock;
  size_t alloc_trace_next = 0;
  std::vector<T>*
      alloc_trace; // pointer because we need to intentionally leak this on
                   // deallocation it can hold references to Python state which
                   // will already be destroyed when we are in exit handlers
};
} // anonymous namespace
} // namespace Native

static std::string reportProcessMemoryInfo(const cudaDeviceProp& prop) {
#ifdef PYTORCH_C10_DRIVER_API_SUPPORTED
  void* nvml_handle = DriverAPI::get_nvml_handle();
  if (!nvml_handle) {
    return "";
  }
  static bool nvml_init [[maybe_unused]] = []() {
    TORCH_INTERNAL_ASSERT(NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_());
    return true;
  }();

  // NOLINTNEXTLINE(*-c-arrays)
  char pci_id[80];
  snprintf(
      pci_id,
      sizeof(pci_id),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);

  nvmlDevice_t nvml_device = nullptr;
  TORCH_INTERNAL_ASSERT(
      NVML_SUCCESS ==
      DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_(
          pci_id, &nvml_device));

  std::vector<nvmlProcessInfo_v1_t> procs(8);
  unsigned int size = procs.size();
  nvmlReturn_t r{};
  while ((r = DriverAPI::get()->nvmlDeviceGetComputeRunningProcesses_(
              nvml_device, &size, procs.data())) ==
         NVML_ERROR_INSUFFICIENT_SIZE) {
    procs.resize(size);
  }
  unsigned int self_pid = getpid();
  std::stringstream ss;
  TORCH_INTERNAL_ASSERT(NVML_SUCCESS == r);
  ss << "";
  for (auto i : c10::irange(size)) {
    auto& proc = procs[i];
    if (self_pid == proc.pid) {
      ss << "Including non-PyTorch memory, this process";
    } else {
      ss << "Process " << proc.pid;
    }
    ss << " has " << format_size(proc.usedGpuMemory) << " memory in use. ";
  }
  return ss.str();
#else
  return "";
#endif
}

namespace Native {

class DeviceCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  c10::DeviceIndex device_id;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks;

  // captures_underway tracks if we are diverting some
  // allocations to a specific pool.
  // Most of the time it's empty, in which case malloc can avoid calling
  // cudaStreamGetCaptureInfo in the hot path.
  std::vector<std::pair<MempoolId_t, std::function<bool(cudaStream_t)>>>
      captures_underway;

  // tracks which pools we can use as a last resort before ooming
  ska::flat_hash_set<MempoolId_t, MempoolIdHash> use_on_oom_pools;

  // Map of blocks whose freeing is deferred until after CUDA graph capture.
  //   - Key: Block* to be freed.
  //   - Value: List of "empty nodes" inserted as free markers during capture.
  //     If the vector is empty, the block must always be deferred until capture
  //     ends.
  ska::flat_hash_map<Block*, std::vector<cudaGraphNode_t>> deferred_blocks;

  // Incremental reverse-traversal state cached per graph.
  // We never re-traverse nodes we've already seen
  struct GraphReuseContext {
    ska::flat_hash_map<cudaStream_t, ska::flat_hash_set<cudaGraphNode_t>>
        visited;
  };
  ska::flat_hash_map<MempoolId_t, CaptureId_t, MempoolIdHash>
      mempool_to_capture_id;
  ska::flat_hash_map<CaptureId_t, GraphReuseContext> graph_reuse_context;

  // outstanding cuda events
  ska::flat_hash_map<
      cuda::CUDAStream,
      std::deque<std::pair<EventPool::Event, Block*>>>
      cuda_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  cudaDeviceProp device_prop;

  // maximum amount of memory that device is allowed to
  // allocate. This is set iff memory fraction is less than 1
  std::optional<size_t> allowed_memory_maximum{std::nullopt};

  // all live expandable segments
  std::vector<ExpandableSegment*> expandable_segments_;
  std::vector<c10::DeviceIndex> devices_with_peer_access_;

  bool record_history = false;

  std::atomic<CreateContextFn> context_recorder_;
  RecordContext record_context_ = RecordContext::NEVER;

  // Ring buffer for memory snapshot TraceEntry's
  RingBuffer<TraceEntry> alloc_buffer;

  // Members specific to CUDA graphs

  // Private pools for CUDA graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

  std::vector<AllocatorTraceTracker> trace_trackers_;

  // mapping from block to a stream_set, containing streams on which the block
  // was used while cudagraph capturing
  std::unordered_map<Block*, stream_set> block_to_cudagraph_stream_uses;

  // thread local compile context for each device
  static thread_local std::stack<std::string> compile_context;

  // thread local user metadata for annotating allocations
  static thread_local std::string user_metadata;

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit DeviceCachingAllocator(c10::DeviceIndex id)
      : device_id(id),
        large_blocks(/*small=*/false),
        small_blocks(/*small=*/true) {
    C10_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, id));

    setMemoryFraction(CUDAAllocatorConfig::per_process_memory_fraction());
    stats.max_split_size =
        static_cast<int64_t>(AcceleratorAllocatorConfig::max_split_size());
    context_recorder_.store(nullptr);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_buffer_max_entries,
      RecordContext when,
      bool clearHistory) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(when == RecordContext::NEVER || context_recorder);
    record_history = enabled;
    context_recorder_.store(record_history ? context_recorder : nullptr);
    alloc_buffer.setMaxEntries(alloc_buffer_max_entries);
    record_context_ = enabled ? when : RecordContext::NEVER;
    if (!enabled || clearHistory) {
      alloc_buffer.clear();
    }
  }

  bool isHistoryEnabled() const {
    return record_history;
  }

  void pushCompileContext(std::string& md) {
    compile_context.push(md);
  }

  void popCompileContext() {
    if (!compile_context.empty()) {
      compile_context.pop();
    }
  }

  void setUserMetadata(const std::string& metadata) {
    user_metadata = metadata;
  }

  std::string getUserMetadata() {
    return user_metadata;
  }

  bool checkPoolLiveAllocations(
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) const {
    std::unique_lock<std::recursive_mutex> lock(mutex);

    PrivatePool* pool = nullptr;
    auto pool_it = graph_pools.find(mempool_id);
    TORCH_CHECK(pool_it != graph_pools.end(), "Could not find pool of id");
    pool = pool_it->second.get();

    TORCH_INTERNAL_ASSERT(pool != nullptr);

    size_t allocated_pool_blocks = 0;

    for (Block* b : active_blocks) {
      TORCH_INTERNAL_ASSERT(b != nullptr);
      TORCH_INTERNAL_ASSERT(b->pool != nullptr);
      if (b->allocated && b->pool->owner_PrivatePool == pool) {
        if (!expected_live_allocations.count(b->ptr)) {
          return false;
        }

        allocated_pool_blocks += 1;
      }
    }

    return allocated_pool_blocks == expected_live_allocations.size();
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
    oom_observers_.emplace_back(std::move(observer));
  }

  void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    trace_trackers_.emplace_back(std::move(tracker));
  }

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<GatheredContext> maybeGatherContext(RecordContext level) {
    if (record_context_ < level) {
      return nullptr;
    }
    return context_recorder_.load()();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(size_t orig_size, cudaStream_t stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway.empty())) {
      // Processes end-of-life events for outstanding allocations used on
      // multiple streams (checks if their GPU-side uses are complete and
      // recycles their memory if so)
      //
      // Q. Why skip process_events if a capture might be underway?
      // A. process_events involves cudaEventQueries, illegal during CUDA graph
      //    capture.
      //    Dumb simple solution: defer reclaiming these allocations until after
      //    capture. Cross-stream memory use is uncommon, so the deferral's
      //    effect on memory use during capture should be small.
      process_events(context);
    } else {
      if (CUDAAllocatorConfig::graph_capture_record_stream_reuse()) {
```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 300 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cuda`, `c10`, `Native`, `CudaMallocAsync`

**Classes/Structs**: `CUmemFabricHandle_st`, `Block`, `PrivatePool`, `BlockPool`, `ExpandableSegment`, `Block`, `SegmentRange`, `ExpandableSegment`, `Handle`, `ShareHeader`, `ExpandableSegment`, `a`, `BlockState`, `SegmentState`, `PrivatePoolState`, `RestoreResult`, `AllocParams`, `EventPool`, `PerDevicePool`, `PrivatePool`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/cuda`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/cuda/CUDACachingAllocator.h`
- `c10/core/impl/GPUTrace.h`
- `c10/cuda/CUDAAllocatorConfig.h`
- `c10/cuda/CUDAException.h`
- `c10/cuda/CUDAFunctions.h`
- `c10/cuda/CUDAGuard.h`
- `c10/util/Gauge.h`
- `c10/util/Logging.h`
- `c10/util/ScopeExit.h`
- `c10/util/UniqueVoidPtr.h`
- `c10/util/env.h`
- `c10/util/error.h`
- `c10/util/flat_hash_map.h`
- `c10/util/hash.h`
- `c10/util/llvmMathExtras.h`
- `c10/util/static_tracepoint.h`
- `c10/cuda/driver_api.h`
- `sys/syscall.h`
- `sys/types.h`
- `unistd.h`
- `c10/util/Exception.h`
- `cuda_runtime_api.h`
- `algorithm`
- `cstddef`
- `cstdint`
- `deque`
- `memory`
- `mutex`
- `new`
- `regex`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`c10/cuda`):

- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`CUDACachingAllocator.h_docs.md`](./CUDACachingAllocator.h_docs.md)
- [`CUDAAlgorithm.h_docs.md`](./CUDAAlgorithm.h_docs.md)
- [`CUDAFunctions.h_docs.md`](./CUDAFunctions.h_docs.md)
- [`CUDAAllocatorConfig.cpp_docs.md`](./CUDAAllocatorConfig.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`CUDAMallocAsyncAllocator.cpp_docs.md`](./CUDAMallocAsyncAllocator.cpp_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`CUDAException.h_docs.md`](./CUDAException.h_docs.md)


## Cross-References

- **File Documentation**: `CUDACachingAllocator.cpp_docs.md`
- **Keyword Index**: `CUDACachingAllocator.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
