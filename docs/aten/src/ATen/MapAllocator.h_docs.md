# Documentation: `aten/src/ATen/MapAllocator.h`

## File Metadata

- **Path**: `aten/src/ATen/MapAllocator.h`
- **Size**: 3,688 bytes (3.60 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Allocator.h>
#include <string_view>

namespace at {

enum MappedAllocatorModes {
  ALLOCATOR_MAPPED_SHARED = 1,
  ALLOCATOR_MAPPED_SHAREDMEM = 2,
  ALLOCATOR_MAPPED_EXCLUSIVE = 4,
  ALLOCATOR_MAPPED_NOCREATE = 8,
  ALLOCATOR_MAPPED_KEEPFD = 16,
  ALLOCATOR_MAPPED_FROMFD = 32,
  ALLOCATOR_MAPPED_UNLINK = 64
};

// Sentinel value/type to help distinguish the file descriptor constructor from
// the non-file descriptor constructor
enum WithFd { WITH_FD };

TORCH_API std::string NewProcessWideShmHandle();

class TORCH_API MapAllocator {
 public:
  MapAllocator(std::string_view filename, int flags, size_t size);
  MapAllocator(
      WithFd /*unused*/,
      std::string_view filename,
      int fd,
      int flags,
      size_t size);
  MapAllocator(const MapAllocator&) = delete;
  MapAllocator& operator=(const MapAllocator&) = delete;
  MapAllocator(MapAllocator&&) = delete;
  MapAllocator& operator=(MapAllocator&&) = delete;

  const char* filename() const {
    return filename_.c_str();
  }
  int fd() const {
#ifdef _WIN32
    TORCH_CHECK(false, "MapAllocator::fd() is unsupported on Windows");
#else
    return fd_;
#endif
  }
  ptrdiff_t size() const {
    return size_;
  }
  // Return a pointer to the actual data for this allocator
  // (in the case of the refcounted allocator, this is offset
  // from the base pointer.)
  virtual void* data() const {
    return base_ptr_;
  }

  int flags() const {
    return flags_;
  }

  static MapAllocator* fromDataPtr(const at::DataPtr& /*dptr*/);
  static at::DataPtr makeDataPtr(
      std::string_view filename,
      int flags,
      size_t size,
      size_t* actual_size_out);
  static at::DataPtr makeDataPtr(
      WithFd /*unused*/,
      const char* filename,
      int fd,
      int flags,
      size_t size,
      size_t* actual_size_out);

  // Closes the data.  Helps us avoid destructor shenanigans
  virtual void close();

  // This is very dangerous.  You have to redefine this destructor for each
  // subclass
  virtual ~MapAllocator();

 protected:
  bool closed_ = false;
  std::string filename_;
  int flags_ = 0;
  ptrdiff_t size_; /* mapped size */
#ifdef _WIN32
  void* handle_;
  void* event_;
  std::string eventname_;
#else
  int fd_ = -1;
#endif
  void* base_ptr_ = nullptr;
};

// Base-from-member idiom
struct TORCH_API RefcountedMapAllocatorArgCheck {
  RefcountedMapAllocatorArgCheck(int flags);
};

class TORCH_API RefcountedMapAllocator : private RefcountedMapAllocatorArgCheck,
                                         public MapAllocator {
 public:
  RefcountedMapAllocator(const char* filename, int flags, size_t size);
  RefcountedMapAllocator(
      WithFd /*unused*/,
      const char* filename,
      int fd,
      int flags,
      size_t size);

  static RefcountedMapAllocator* fromDataPtr(const at::DataPtr& /*dptr*/);
  RefcountedMapAllocator(const RefcountedMapAllocator&) = delete;
  RefcountedMapAllocator(RefcountedMapAllocator&&) = delete;
  RefcountedMapAllocator& operator=(const RefcountedMapAllocator&) = delete;
  RefcountedMapAllocator& operator=(RefcountedMapAllocator&&) = delete;
  static at::DataPtr makeDataPtr(
      const char* filename,
      int flags,
      size_t size,
      size_t* actual_size_out);
  static at::DataPtr makeDataPtr(
      WithFd /*unused*/,
      const char* filename,
      int fd,
      int flags,
      size_t size,
      size_t* actual_size_out);

  void* data() const override;

  void incref();
  int decref();
  void close() override;

  ~RefcountedMapAllocator() override {
    RefcountedMapAllocator::close();
  }

 protected:
  void checkFlags();
  void initializeAlloc();
};

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TORCH_API`, `virtual`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Allocator.h`
- `string_view`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `MapAllocator.h_docs.md`
- **Keyword Index**: `MapAllocator.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
