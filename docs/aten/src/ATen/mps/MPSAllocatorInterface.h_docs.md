# Documentation: `aten/src/ATen/mps/MPSAllocatorInterface.h`

## File Metadata

- **Path**: `aten/src/ATen/mps/MPSAllocatorInterface.h`
- **Size**: 2,718 bytes (2.65 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
//  Copyright © 2023 Apple Inc.

#pragma once

#include <ATen/core/ATen_fwd.h>
#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>

#define MB(x) (x * 1048576UL)

namespace at::mps {

// this is a public interface to access MPSAllocator.
// Do not declare methods that would depend on MPS or Metal frameworks.
class IMPSAllocator : public c10::Allocator {
 public:
  // see the comments in MPSAllocator.h for the description of these methods.
  virtual void emptyCache() const = 0;
  virtual void freeInactiveBuffers() const = 0;
  virtual ssize_t getUnalignedBufferSize(const void* ptr) const = 0;
  virtual IntArrayRef getBufferShape(const void* ptr) const = 0;
  virtual id_t getBufferId(const void* ptr) const = 0;
  virtual void setBufferShape(const void* ptr, const IntArrayRef& shape)
      const = 0;
  virtual bool isSharedBuffer(const void* ptr) const = 0;
  virtual bool isSharedStorageSupported() const = 0;
  virtual c10::DataPtr allocScalarBufferWithValue(void* value, size_t size)
      const = 0;
  virtual std::string formatSize(size_t size) const = 0;
  virtual void setLowWatermarkRatio(double ratio) const = 0;
  virtual void setHighWatermarkRatio(double ratio) const = 0;
  virtual ssize_t getLowWatermarkValue() const = 0;
  virtual size_t getLowWatermarkLimit() const = 0;
  virtual size_t getHighWatermarkLimit() const = 0;
  virtual size_t getTotalAllocatedMemory() const = 0;
  virtual size_t getCurrentAllocatedMemory() const = 0;
  virtual size_t getDriverAllocatedMemory() const = 0;
  virtual size_t getRecommendedMaxMemory() const = 0;
  virtual std::pair<const void*, uint32_t> getSharedBufferPtr(
      const void* ptr) const = 0;
  virtual bool recordEvents(c10::ArrayRef<const void*> buffers) const = 0;
  virtual bool waitForEvents(c10::ArrayRef<const void*> buffers) const = 0;
};

class IMpsAllocatorCallback {
 public:
  enum class EventType {
    ALLOCATED, // buffer got allocated to be used immediately
    RECYCLED, // buffer pulled from free list to be reused
    FREED, // buffer put to free list for future recycling
    RELEASED, // buffer memory released
    ALLOCATION_FAILED // buffer allocation failed
  };
  virtual ~IMpsAllocatorCallback() = default;
  virtual void executeMPSAllocatorCallback(void* ptr, EventType event) = 0;
};

// MPS allocator will execute every registered callback when a block of memory
// is freed.
TORCH_DECLARE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);
#define REGISTER_MPS_ALLOCATOR_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(MPSAllocatorCallbacksRegistry, name, __VA_ARGS__)

IMPSAllocator* getIMPSAllocator(bool sharedAllocator = false);

bool isMPSPinnedPtr(const void* data);

} // namespace at::mps

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 25 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `IMPSAllocator`, `IMpsAllocatorCallback`, `EventType`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/mps`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ATen_fwd.h`
- `c10/core/Allocator.h`
- `c10/util/Registry.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`aten/src/ATen/mps`):

- [`MPSProfiler.h_docs.md`](./MPSProfiler.h_docs.md)
- [`MPSAllocator.h_docs.md`](./MPSAllocator.h_docs.md)
- [`MPSDevice.h_docs.md`](./MPSDevice.h_docs.md)
- [`MPSEvent.h_docs.md`](./MPSEvent.h_docs.md)
- [`MPSGuardImpl.h_docs.md`](./MPSGuardImpl.h_docs.md)
- [`MPSHooks.h_docs.md`](./MPSHooks.h_docs.md)
- [`EmptyTensor.h_docs.md`](./EmptyTensor.h_docs.md)
- [`IndexKernels.h_docs.md`](./IndexKernels.h_docs.md)
- [`MPSGeneratorImpl.h_docs.md`](./MPSGeneratorImpl.h_docs.md)


## Cross-References

- **File Documentation**: `MPSAllocatorInterface.h_docs.md`
- **Keyword Index**: `MPSAllocatorInterface.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
