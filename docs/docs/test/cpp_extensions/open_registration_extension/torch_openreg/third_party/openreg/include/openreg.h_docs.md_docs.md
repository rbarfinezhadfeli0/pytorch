# Documentation: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include/openreg.h_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include/openreg.h_docs.md`
- **Size**: 5,237 bytes (5.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include/openreg.h`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include/openreg.h`
- **Size**: 3,095 bytes (3.02 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```c
#pragma once

#include <cstddef>

#ifdef _WIN32
#define OPENREG_EXPORT __declspec(dllexport)
#else
#define OPENREG_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum orError_t {
  orSuccess = 0,
  orErrorUnknown = 1,
  orErrorNotReady = 2
} orError_t;

typedef enum orMemcpyKind {
  orMemcpyHostToHost = 0,
  orMemcpyHostToDevice = 1,
  orMemcpyDeviceToHost = 2,
  orMemcpyDeviceToDevice = 3
} orMemcpyKind;

typedef enum orMemoryType {
  orMemoryTypeUnmanaged = 0,
  orMemoryTypeHost = 1,
  orMemoryTypeDevice = 2
} orMemoryType;

struct orPointerAttributes {
  orMemoryType type = orMemoryType::orMemoryTypeUnmanaged;
  int device;
  void* pointer;
};

typedef enum orEventFlags {
  orEventDisableTiming = 0x0,
  orEventEnableTiming = 0x1,
} orEventFlags;

struct orStream;
struct orEvent;
typedef struct orStream* orStream_t;
typedef struct orEvent* orEvent_t;

// Memory
OPENREG_EXPORT orError_t orMalloc(void** devPtr, size_t size);
OPENREG_EXPORT orError_t orFree(void* devPtr);
OPENREG_EXPORT orError_t orMallocHost(void** hostPtr, size_t size);
OPENREG_EXPORT orError_t orFreeHost(void* hostPtr);
OPENREG_EXPORT orError_t
orMemcpy(void* dst, const void* src, size_t count, orMemcpyKind kind);
OPENREG_EXPORT orError_t orMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind,
    orStream_t stream);
OPENREG_EXPORT orError_t
orPointerGetAttributes(orPointerAttributes* attributes, const void* ptr);
OPENREG_EXPORT orError_t orMemoryUnprotect(void* devPtr);
OPENREG_EXPORT orError_t orMemoryProtect(void* devPtr);

// Device
OPENREG_EXPORT orError_t orGetDeviceCount(int* count);
OPENREG_EXPORT orError_t orSetDevice(int device);
OPENREG_EXPORT orError_t orGetDevice(int* device);
OPENREG_EXPORT orError_t
orDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
OPENREG_EXPORT orError_t orDeviceSynchronize(void);

// Stream
OPENREG_EXPORT orError_t orStreamCreateWithPriority(
    orStream_t* stream,
    unsigned int flags,
    int priority);
OPENREG_EXPORT orError_t orStreamCreate(orStream_t* stream);
OPENREG_EXPORT orError_t orStreamGetPriority(orStream_t stream, int* priority);
OPENREG_EXPORT orError_t orStreamDestroy(orStream_t stream);
OPENREG_EXPORT orError_t orStreamQuery(orStream_t stream);
OPENREG_EXPORT orError_t orStreamSynchronize(orStream_t stream);
OPENREG_EXPORT orError_t
orStreamWaitEvent(orStream_t stream, orEvent_t event, unsigned int flags);

// Event
OPENREG_EXPORT orError_t
orEventCreateWithFlags(orEvent_t* event, unsigned int flags);
OPENREG_EXPORT orError_t orEventCreate(orEvent_t* event);
OPENREG_EXPORT orError_t orEventDestroy(orEvent_t event);
OPENREG_EXPORT orError_t orEventRecord(orEvent_t event, orStream_t stream);
OPENREG_EXPORT orError_t orEventSynchronize(orEvent_t event);
OPENREG_EXPORT orError_t orEventQuery(orEvent_t event);
OPENREG_EXPORT orError_t
orEventElapsedTime(float* ms, orEvent_t start, orEvent_t end);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus

#define OPENREG_H
#include "openreg.inl"

#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 30 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `orPointerAttributes`, `orStream`, `orEvent`, `orStream`, `orEvent`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `cstddef`
- `openreg.inl`


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

This is a test file. Run it with:

```bash
python test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include/openreg.h
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include`):



## Cross-References

- **File Documentation**: `openreg.h_docs.md`
- **Keyword Index**: `openreg.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include/openreg.h_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include`):

- [`openreg.h_kw.md_docs.md`](./openreg.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `openreg.h_docs.md_docs.md`
- **Keyword Index**: `openreg.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
