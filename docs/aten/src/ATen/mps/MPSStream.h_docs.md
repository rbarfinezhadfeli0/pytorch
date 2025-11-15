# Documentation: `aten/src/ATen/mps/MPSStream.h`

## File Metadata

- **Path**: `aten/src/ATen/mps/MPSStream.h`
- **Size**: 4,840 bytes (4.73 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
//  Copyright © 2022 Apple Inc.

#pragma once

#include <cstdint>
#include <utility>

#include <ATen/mps/MPSDevice.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
typedef MPSCommandBuffer* MPSCommandBuffer_t;
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef id<MTLSharedEvent> MTLSharedEvent_t;
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLBuffer> MTLBuffer_t;
#else
#include <dispatch/dispatch.h>
typedef void* MPSCommandBuffer_t;
typedef void* MPSGraph;
typedef void* MPSGraphExecutionDescriptor;
typedef void* MPSGraphCompilationDescriptor;
typedef void* MTLCommandQueue_t;
typedef void* MTLComputeCommandEncoder_t;
typedef void* MTLSharedEvent_t;
typedef void* MTLDevice_t;
typedef void* MTLBuffer_t;
typedef void* MTLCommandBufferHandler;
typedef void* NSDictionary;
#define nil NULL
#endif

namespace at::mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

enum class SyncType {
  NONE, // no commit to command buffer
  COMMIT, // commit and flush the command buffer
  COMMIT_AND_WAIT, // flush and wait for command buffer execution to finish
  COMMIT_AND_CONTINUE, // commit and continue with a new underlying command buffer
  COMMIT_ADAPTIVE, // commit adaptively based on available memory
};

class TORCH_API MPSStream {
 public:
  enum Unchecked { UNCHECKED };

  /// Construct a MPSStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a MPS stream.
  explicit MPSStream(Stream stream);

  ~MPSStream();

  MTLCommandQueue_t commandQueue() const {
    return _commandQueue;
  }

  dispatch_queue_t queue() const {
    return _serialQueue;
  }

  MPSCommandBuffer_t commandBuffer();
  MTLComputeCommandEncoder_t commandEncoder();
  void endKernelCoalescing();
  void synchronize(SyncType syncType);
  void fill(MTLBuffer_t buffer, uint8_t value, size_t length, size_t offset, SyncType syncType = SyncType::NONE);
  void copy(MTLBuffer_t srcBuffer,
            MTLBuffer_t dstBuffer,
            size_t length,
            size_t srcOffset,
            size_t dstOffset,
            uint64_t profileId,
            SyncType syncType = SyncType::NONE);
  void copy_and_sync(MTLBuffer_t srcBuffer,
                     MTLBuffer_t dstBuffer,
                     size_t length,
                     size_t srcOffset,
                     size_t dstOffset,
                     bool non_blocking,
                     uint64_t profileId);
  void executeMPSGraph(MPSGraph* mpsGraph,
                       NSDictionary* feeds,
                       NSDictionary* results,
                       SyncType syncType = SyncType::NONE);
  void addCompletedHandler(MTLCommandBufferHandler block);

  /// Get the MPS device index that this stream is associated with.
  c10::DeviceIndex device_index() const {
    return _stream.device_index();
  }

  MTLCommandQueue_t stream() const {
    return _commandQueue;
  }

  MTLDevice_t device() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const {
    return _stream;
  }

  MTLBuffer_t getErrorBuffer();
  void checkLastError();

 private:
  Stream _stream;
  MTLCommandQueue_t _commandQueue = nil;
  MPSCommandBuffer_t _commandBuffer = nil;
  MPSCommandBuffer_t _prevCommandBuffer = nil;
  MTLComputeCommandEncoder_t _commandEncoder = nil;
  MPSGraphExecutionDescriptor* _executionDescriptor = nil;
  MPSGraphCompilationDescriptor* _compilationDescriptor = nil;
  dispatch_queue_t _serialQueue = nullptr;
  // CommitAndContinue is enabled by default
  bool _enableCommitAndContinue = true;
  // Buffer that contains last raised error
  MTLBuffer_t _errorBuffer = nil;

  // use synchronize() to access any of these commit functions outside MPSStream
  void commit();
  void commitAndWait();
  void commitAndContinue();
  void flush();
};

/**
 * Get the current MPS stream
 */
TORCH_API MPSStream* getCurrentMPSStream();

/**
 * Get the default MPS stream
 */
TORCH_API MPSStream* getDefaultMPSStream();

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

class TORCH_API MPSStreamImpl {
 public:
  /**
   * Gets single instance of the MPSStream.
   */
  static MPSStream* getInstance();

 private:
  static MPSStream* _stream;
  MPSStreamImpl();
};

#ifdef __OBJC__
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());
#endif
} // namespace at::mps

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 24 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `SyncType`, `TORCH_API`, `a`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/mps`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `utility`
- `ATen/mps/MPSDevice.h`
- `c10/core/DeviceGuard.h`
- `c10/core/Stream.h`
- `c10/util/Exception.h`
- `Foundation/Foundation.h`
- `Metal/Metal.h`
- `MetalPerformanceShaders/MetalPerformanceShaders.h`
- `MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h`
- `dispatch/dispatch.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`aten/src/ATen/mps`):

- [`MPSProfiler.h_docs.md`](./MPSProfiler.h_docs.md)
- [`MPSAllocator.h_docs.md`](./MPSAllocator.h_docs.md)
- [`MPSDevice.h_docs.md`](./MPSDevice.h_docs.md)
- [`MPSAllocatorInterface.h_docs.md`](./MPSAllocatorInterface.h_docs.md)
- [`MPSEvent.h_docs.md`](./MPSEvent.h_docs.md)
- [`MPSGuardImpl.h_docs.md`](./MPSGuardImpl.h_docs.md)
- [`MPSHooks.h_docs.md`](./MPSHooks.h_docs.md)
- [`EmptyTensor.h_docs.md`](./EmptyTensor.h_docs.md)
- [`IndexKernels.h_docs.md`](./IndexKernels.h_docs.md)
- [`MPSGeneratorImpl.h_docs.md`](./MPSGeneratorImpl.h_docs.md)


## Cross-References

- **File Documentation**: `MPSStream.h_docs.md`
- **Keyword Index**: `MPSStream.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
