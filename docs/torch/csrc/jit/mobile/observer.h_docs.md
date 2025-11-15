# Documentation: `torch/csrc/jit/mobile/observer.h`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/observer.h`
- **Size**: 3,826 bytes (3.74 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/ThreadLocalDebugInfo.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {

class MobileDebugInfo : public c10::DebugInfoBase {
 public:
  const std::string& getModelName() {
    return model_name_;
  }

  void setModelName(const std::string& model_name) {
    model_name_ = model_name;
  }

  const std::string& getMethodName() {
    return method_name_;
  }

  void setMethodName(const std::string& method_name) {
    method_name_ = method_name;
  }

  size_t getOpIdx() {
    return op_idx_;
  }

  void setOpIdx(size_t op_idx) {
    op_idx_ = op_idx;
  }

 private:
  std::string model_name_;
  std::string method_name_;
  // TODO: Kimish
  // If we launch a thread such as for at::launch, interepter continuation
  // and if the caching allocator is enabled in the base thread
  // then, in order to propagate this information, that is caching allocator
  // is enabled, across thread boundaries we can use the mechanism provided
  // by ThreadLocalDebugInfo
  // Once the thread local MobileDebugInfo is accessible in the launched
  // thread, it can be accessed in that thread and that thread can set
  // its own thread local CachingAllocatorInfo.
  // However, we cannot expect every launched thread to extract and set
  // its own thread local copy of CachingAllocatorInfo.
  // But this can be done in lite interpreter, where in the run method
  // it can do info =
  // c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::MOBILE_RUNTIME_INFO))
  // .get_caching_allocator_info();
  // GetThreadLocalCachingAllocatorInfo() = info;
  // Other option is to have MobileDebugInfo itself be the place where thread
  // local copy of CachingAllocatorInfo is stored. Then
  // DefaultMobileCPUAllocator inspects this to decide if to use
  // CachingAllocator. However, current lite interpreter does not support FORK,
  // thus from the run method of lite interpreter we are not really gonna launch
  // another instance of lite interpreter in a different thread. So for now not
  // getting bothered about passing CachingAllocatorInfo across thread
  // boundaries. c10::CachingAllocatorInfo caching_allocator_info;
  size_t op_idx_ = 0;
};

class MobileModuleObserver {
 public:
  virtual ~MobileModuleObserver() = default;

  virtual void onEnterRunMethod(const int32_t /*unused*/) {}
  virtual void onExitRunMethod(
      const std::unordered_map<std::string, std::string>& /*unused*/,
      const std::string& /*unused*/,
      const int32_t /*unused*/) {}
  virtual void onFailRunMethod(
      const std::unordered_map<std::string, std::string>& /*unused*/,
      const std::string& /*unused*/,
      const int32_t /*unused*/,
      const char* /*unused*/) {}
  virtual void onEnterLoadModel(const int32_t /*unused*/) {}
  virtual void onExitLoadModel(
      const int32_t /*unused*/,
      const std::unordered_map<std::string, std::string>& /*unused*/) {
  } // key: filename, value: file content
  virtual void onFailLoadModel(
      const int32_t /*unused*/,
      const char* /*unused*/) {}
  virtual void onFailLoadModel(
      const int32_t /*unused*/,
      const char* /*unused*/,
      const std::unordered_map<std::string, std::string>& /*unused*/) {}
  virtual std::vector<std::string> getDefaultExtraFiles() = 0;
  virtual std::unordered_map<std::string, std::string> processMetadataFromExtra(
      const std::unordered_map<std::string, std::string>&) = 0;
};

class MobileObserverConfig {
 public:
  void setModuleObserver(std::unique_ptr<MobileModuleObserver> reporter) {
    module_observer_ = std::move(reporter);
  }
  MobileModuleObserver* getModuleObserver() {
    return module_observer_.get();
  }

 private:
  std::unique_ptr<MobileModuleObserver> module_observer_;
};

MobileObserverConfig& observerConfig();

} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `MobileDebugInfo`, `MobileModuleObserver`, `MobileObserverConfig`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/ThreadLocalDebugInfo.h`
- `string`
- `unordered_map`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/csrc/jit/mobile`):

- [`register_ops_common_utils.cpp_docs.md`](./register_ops_common_utils.cpp_docs.md)
- [`import.h_docs.md`](./import.h_docs.md)
- [`prim_ops_registery.h_docs.md`](./prim_ops_registery.h_docs.md)
- [`profiler_edge.h_docs.md`](./profiler_edge.h_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`file_format.h_docs.md`](./file_format.h_docs.md)
- [`module.h_docs.md`](./module.h_docs.md)
- [`module.cpp_docs.md`](./module.cpp_docs.md)
- [`flatbuffer_loader.cpp_docs.md`](./flatbuffer_loader.cpp_docs.md)


## Cross-References

- **File Documentation**: `observer.h_docs.md`
- **Keyword Index**: `observer.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
