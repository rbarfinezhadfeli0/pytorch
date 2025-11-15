# Documentation: `c10/util/ThreadLocalDebugInfo.h`

## File Metadata

- **Path**: `c10/util/ThreadLocalDebugInfo.h`
- **Size**: 2,665 bytes (2.60 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/macros/Export.h>

#include <cstdint>
#include <memory>

namespace c10 {

enum class C10_API_ENUM DebugInfoKind : uint8_t {
  PRODUCER_INFO = 0,
  MOBILE_RUNTIME_INFO,
  PROFILER_STATE,
  INFERENCE_CONTEXT, // for inference usage
  PARAM_COMMS_INFO,

  TEST_INFO, // used only in tests
  TEST_INFO_2, // used only in tests
};

class C10_API DebugInfoBase {
 public:
  DebugInfoBase() = default;
  virtual ~DebugInfoBase() = default;
};

// Thread local debug information is propagated across the forward
// (including async fork tasks) and backward passes and is supposed
// to be utilized by the user's code to pass extra information from
// the higher layers (e.g. model id) down to the lower levels
// (e.g. to the operator observers used for debugging, logging,
// profiling, etc)
class C10_API ThreadLocalDebugInfo {
 public:
  static DebugInfoBase* get(DebugInfoKind kind);

  // Get current ThreadLocalDebugInfo
  static std::shared_ptr<ThreadLocalDebugInfo> current();

  // Internal, use DebugInfoGuard/ThreadLocalStateGuard
  static void _forceCurrentDebugInfo(
      std::shared_ptr<ThreadLocalDebugInfo> info);

  // Push debug info struct of a given kind
  static void _push(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);
  // Pop debug info, throws in case the last pushed
  // debug info is not of a given kind
  static std::shared_ptr<DebugInfoBase> _pop(DebugInfoKind kind);
  // Peek debug info, throws in case the last pushed debug info is not of the
  // given kind
  static std::shared_ptr<DebugInfoBase> _peek(DebugInfoKind kind);

 private:
  std::shared_ptr<DebugInfoBase> info_;
  DebugInfoKind kind_;
  std::shared_ptr<ThreadLocalDebugInfo> parent_info_;

  friend class DebugInfoGuard;
};

// DebugInfoGuard is used to set debug information,
// ThreadLocalDebugInfo is semantically immutable, the values are set
// through the scope-based guard object.
// Nested DebugInfoGuard adds/overrides existing values in the scope,
// restoring the original values after exiting the scope.
// Users can access the values through the ThreadLocalDebugInfo::get() call;
class C10_API DebugInfoGuard {
 public:
  DebugInfoGuard(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

  explicit DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfo> info);

  ~DebugInfoGuard();

  DebugInfoGuard(const DebugInfoGuard&) = delete;
  DebugInfoGuard(DebugInfoGuard&&) = delete;
  DebugInfoGuard& operator=(const DebugInfoGuard&) = delete;
  DebugInfoGuard& operator=(DebugInfoGuard&&) = delete;

 private:
  bool active_ = false;
  std::shared_ptr<ThreadLocalDebugInfo> prev_info_ = nullptr;
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `C10_API_ENUM`, `C10_API`, `C10_API`, `of`, `DebugInfoGuard`, `C10_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Export.h`
- `cstdint`
- `memory`


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

Files in the same folder (`c10/util`):

- [`CallOnce.h_docs.md`](./CallOnce.h_docs.md)
- [`Unicode.cpp_docs.md`](./Unicode.cpp_docs.md)
- [`logging_is_not_google_glog.h_docs.md`](./logging_is_not_google_glog.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`complex_math.h_docs.md`](./complex_math.h_docs.md)
- [`order_preserving_flat_hash_map.h_docs.md`](./order_preserving_flat_hash_map.h_docs.md)
- [`flags_use_gflags.cpp_docs.md`](./flags_use_gflags.cpp_docs.md)
- [`flags_use_no_gflags.cpp_docs.md`](./flags_use_no_gflags.cpp_docs.md)
- [`Float8_e4m3fnuz.h_docs.md`](./Float8_e4m3fnuz.h_docs.md)
- [`typeid.cpp_docs.md`](./typeid.cpp_docs.md)


## Cross-References

- **File Documentation**: `ThreadLocalDebugInfo.h_docs.md`
- **Keyword Index**: `ThreadLocalDebugInfo.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
