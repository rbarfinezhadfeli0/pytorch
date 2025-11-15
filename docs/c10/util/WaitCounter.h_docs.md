# Documentation: `c10/util/WaitCounter.h`

## File Metadata

- **Path**: `c10/util/WaitCounter.h`
- **Size**: 2,620 bytes (2.56 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <chrono>
#include <memory>
#include <string_view>
#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/SmallVector.h>

namespace c10::monitor {
namespace detail {
class WaitCounterImpl;

class WaitCounterBackendIf {
 public:
  virtual ~WaitCounterBackendIf() = default;

  virtual intptr_t start(
      std::chrono::steady_clock::time_point now) noexcept = 0;
  virtual void stop(
      std::chrono::steady_clock::time_point now,
      intptr_t ctx) noexcept = 0;
};

class WaitCounterBackendFactoryIf {
 public:
  virtual ~WaitCounterBackendFactoryIf() = default;

  // May return nullptr.
  // In this case the counter will be ignored by the given backend.
  virtual std::unique_ptr<WaitCounterBackendIf> create(
      std::string_view key) noexcept = 0;
};

C10_API void registerWaitCounterBackend(
    std::unique_ptr<WaitCounterBackendFactoryIf> /*factory*/);

C10_API std::vector<std::shared_ptr<WaitCounterBackendFactoryIf>>
getRegisteredWaitCounterBackends();
} // namespace detail

// A handle to a wait counter.
class C10_API WaitCounterHandle {
 public:
  explicit WaitCounterHandle(std::string_view key);

  class WaitGuard {
   public:
    WaitGuard(WaitGuard&& other) noexcept
        : handle_{std::exchange(other.handle_, {})},
          ctxs_{std::move(other.ctxs_)} {}
    WaitGuard(const WaitGuard&) = delete;
    WaitGuard& operator=(const WaitGuard&) = delete;
    WaitGuard& operator=(WaitGuard&&) = delete;

    ~WaitGuard() {
      stop();
    }

    void stop() {
      if (auto handle = std::exchange(handle_, nullptr)) {
        handle->stop(ctxs_);
      }
    }

   private:
    WaitGuard(WaitCounterHandle& handle, SmallVector<intptr_t>&& ctxs)
        : handle_{&handle}, ctxs_{std::move(ctxs)} {}

    friend class WaitCounterHandle;

    WaitCounterHandle* handle_;
    SmallVector<intptr_t> ctxs_;
  };

  // Starts a waiter
  WaitGuard start();

 private:
  // Stops the waiter. Each start() call should be matched by exactly one stop()
  // call.
  void stop(const SmallVector<intptr_t>& ctxs);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  detail::WaitCounterImpl& impl_;
};
} // namespace c10::monitor

#define STATIC_WAIT_COUNTER(_key)                           \
  []() -> ::c10::monitor::WaitCounterHandle& {              \
    static ::c10::monitor::WaitCounterHandle handle(#_key); \
    return handle;                                          \
  }()

#define STATIC_SCOPED_WAIT_COUNTER(_name) \
  auto C10_ANONYMOUS_VARIABLE(SCOPE_GUARD) = STATIC_WAIT_COUNTER(_name).start();

```



## High-Level Overview


This C++ file contains approximately 6 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `WaitCounterImpl`, `WaitCounterBackendIf`, `WaitCounterBackendFactoryIf`, `C10_API`, `WaitGuard`, `WaitCounterHandle`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `chrono`
- `memory`
- `string_view`
- `vector`
- `c10/macros/Macros.h`
- `c10/util/ScopeExit.h`
- `c10/util/SmallVector.h`


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

- **File Documentation**: `WaitCounter.h_docs.md`
- **Keyword Index**: `WaitCounter.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
