# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/csrc/memory.cpp`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/csrc/memory.cpp`
- **Size**: 6,772 bytes (6.61 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include "memory.h"

#include <include/openreg.h>

#include <map>
#include <mutex>

namespace {

struct Block {
  orMemoryType type = orMemoryType::orMemoryTypeUnmanaged;
  int device = -1;
  void* pointer = nullptr;
  size_t size = 0;
  int refcount{0};
};

class MemoryManager {
 public:
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  orError_t allocate(void** ptr, size_t size, orMemoryType type) {
    if (!ptr || size == 0)
      return orErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    long page_size = openreg::get_pagesize();
    size_t aligned_size = ((size - 1) / page_size + 1) * page_size;
    void* mem = nullptr;
    int current_device = -1;

    if (type == orMemoryType::orMemoryTypeDevice) {
      orGetDevice(&current_device);

      mem = openreg::mmap(aligned_size);
      if (mem == nullptr)
        return orErrorUnknown;
      if (openreg::mprotect(mem, aligned_size, F_PROT_NONE) != 0) {
        openreg::munmap(mem, aligned_size);
        return orErrorUnknown;
      }
    } else {
      if (openreg::alloc(&mem, page_size, aligned_size) != 0) {
        return orErrorUnknown;
      }
    }

    m_registry[mem] = {type, current_device, mem, aligned_size, 0};
    *ptr = mem;
    return orSuccess;
  }

  orError_t free(void* ptr) {
    if (!ptr)
      return orSuccess;

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_registry.find(ptr);
    if (it == m_registry.end())
      return orErrorUnknown;

    const auto& info = it->second;
    if (info.type == orMemoryType::orMemoryTypeDevice) {
      openreg::mprotect(info.pointer, info.size, F_PROT_READ | F_PROT_WRITE);
      openreg::munmap(info.pointer, info.size);
    } else {
      openreg::free(info.pointer);
    }

    m_registry.erase(it);
    return orSuccess;
  }

  orError_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      orMemcpyKind kind) {
    if (!dst || !src || count == 0)
      return orErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    Block* dst_info = getBlockInfoNoLock(dst);
    Block* src_info = getBlockInfoNoLock(src);

    switch (kind) {
      case orMemcpyHostToDevice:
        if ((!dst_info || dst_info->type != orMemoryType::orMemoryTypeDevice) ||
            (src_info && src_info->type == orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyDeviceToHost:
        if ((dst_info && dst_info->type == orMemoryType::orMemoryTypeDevice) ||
            (!src_info || src_info->type != orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyDeviceToDevice:
        if ((!dst_info || dst_info->type != orMemoryType::orMemoryTypeDevice) ||
            (!src_info || src_info->type != orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyHostToHost:
        if ((dst_info && dst_info->type == orMemoryType::orMemoryTypeDevice) ||
            (src_info && src_info->type == orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
    }

    unprotectNoLock(dst_info);
    unprotectNoLock(src_info);
    ::memcpy(dst, src, count);
    protectNoLock(dst_info);
    protectNoLock(src_info);

    return orSuccess;
  }

  orError_t getPointerAttributes(
      orPointerAttributes* attributes,
      const void* ptr) {
    if (!attributes || !ptr)
      return orErrorUnknown;

    std ::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);

    if (!info) {
      attributes->type = orMemoryType::orMemoryTypeUnmanaged;
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
    } else {
      attributes->type = info->type;
      attributes->device = info->device;
      attributes->pointer = info->pointer;
    }

    return orSuccess;
  }

  orError_t unprotect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return unprotectNoLock(getBlockInfoNoLock(ptr));
  }

  orError_t protect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return protectNoLock(getBlockInfoNoLock(ptr));
  }

 private:
  MemoryManager() = default;

  orError_t unprotectNoLock(Block* info) {
    if (info && info->type == orMemoryType::orMemoryTypeDevice) {
      if (info->refcount == 0) {
        if (openreg::mprotect(
                info->pointer, info->size, F_PROT_READ | F_PROT_WRITE) != 0) {
          return orErrorUnknown;
        }
      }

      info->refcount++;
    }

    return orSuccess;
  }

  orError_t protectNoLock(Block* info) {
    if (info && info->type == orMemoryType::orMemoryTypeDevice) {
      if (info->refcount == 1) {
        if (openreg::mprotect(info->pointer, info->size, F_PROT_NONE) != 0) {
          return orErrorUnknown;
        }
      }

      info->refcount--;
    }

    return orSuccess;
  }

  Block* getBlockInfoNoLock(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it != m_registry.begin()) {
      --it;
      const char* p_char = static_cast<const char*>(ptr);
      const char* base_char = static_cast<const char*>(it->first);
      if (p_char >= base_char && p_char < (base_char + it->second.size)) {
        return &it->second;
      }
    }

    return nullptr;
  }

  std::map<void*, Block> m_registry;
  std::mutex m_mutex;
};

} // namespace

orError_t orMalloc(void** devPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      devPtr, size, orMemoryType::orMemoryTypeDevice);
}

orError_t orFree(void* devPtr) {
  return MemoryManager::getInstance().free(devPtr);
}

orError_t orMallocHost(void** hostPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      hostPtr, size, orMemoryType::orMemoryTypeHost);
}

orError_t orFreeHost(void* hostPtr) {
  return MemoryManager::getInstance().free(hostPtr);
}

orError_t orMemcpy(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind) {
  return MemoryManager::getInstance().memcpy(dst, src, count, kind);
}

orError_t orMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind,
    orStream_t stream) {
  if (!stream) {
    return orErrorUnknown;
  }

  auto& mm = MemoryManager::getInstance();

  return orLaunchKernel(
      stream, &MemoryManager::memcpy, &mm, dst, src, count, kind);
}

orError_t orPointerGetAttributes(
    orPointerAttributes* attributes,
    const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
}

orError_t orMemoryUnprotect(void* devPtr) {
  return MemoryManager::getInstance().unprotect(devPtr);
}

orError_t orMemoryProtect(void* devPtr) {
  return MemoryManager::getInstance().protect(devPtr);
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `orError_t`

**Classes/Structs**: `Block`, `MemoryManager`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/csrc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `memory.h`
- `include/openreg.h`
- `map`
- `mutex`


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
python test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/csrc/memory.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/csrc`):

- [`stream.cpp_docs.md`](./stream.cpp_docs.md)
- [`memory.h_docs.md`](./memory.h_docs.md)
- [`device.cpp_docs.md`](./device.cpp_docs.md)


## Cross-References

- **File Documentation**: `memory.cpp_docs.md`
- **Keyword Index**: `memory.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
