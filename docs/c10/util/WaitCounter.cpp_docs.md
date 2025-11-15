# Documentation: `c10/util/WaitCounter.cpp`

## File Metadata

- **Path**: `c10/util/WaitCounter.cpp`
- **Size**: 4,806 bytes (4.69 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/WaitCounter.h>

#include <c10/util/Synchronized.h>
#include <c10/util/WaitCounterDynamicBackend.h>

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace c10::monitor {

namespace detail {
namespace {
using WaitCounterBackendFactories =
    std::vector<std::shared_ptr<WaitCounterBackendFactoryIf>>;

Synchronized<WaitCounterBackendFactories>& waitCounterBackendFactories() {
  static auto instance = new Synchronized<WaitCounterBackendFactories>();
  return *instance;
}

class DynamicBackendWrapper : public WaitCounterBackendIf {
 public:
  explicit DynamicBackendWrapper(WaitCounterDynamicBackend impl)
      : impl_{impl} {}

  DynamicBackendWrapper(const DynamicBackendWrapper&) = delete;
  DynamicBackendWrapper(DynamicBackendWrapper&&) = delete;
  DynamicBackendWrapper& operator=(const DynamicBackendWrapper&) = delete;
  DynamicBackendWrapper& operator=(DynamicBackendWrapper&&) = delete;
  ~DynamicBackendWrapper() override {
    impl_.destroy(impl_.self);
  }

  intptr_t start(std::chrono::steady_clock::time_point now) noexcept override {
    return impl_.start(
        impl_.self,
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch())
            .count());
  }

  void stop(std::chrono::steady_clock::time_point now, intptr_t ctx) noexcept
      override {
    impl_.stop(
        impl_.self,
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch())
            .count(),
        ctx);
  }

 private:
  WaitCounterDynamicBackend impl_;
};

std::unique_ptr<WaitCounterBackendIf> getDynamicBackend(std::string_view key) {
  static auto dynamicBackendInit =
      reinterpret_cast<WaitCounterDynamicBackendInit>([]() -> void* {
#ifndef _WIN32
        return dlsym(
            RTLD_DEFAULT,
            std::string(kWaitCounterDynamicBackendInitFn).c_str());
#else
        return nullptr;
#endif
      }());
  if (!dynamicBackendInit) {
    return nullptr;
  }
  WaitCounterDynamicBackend backend;
  dynamicBackendInit(&backend, &key[0], key.size());
  if (!backend.self) {
    return nullptr;
  }
  return std::make_unique<DynamicBackendWrapper>(backend);
}
} // namespace

class WaitCounterImpl {
 public:
  static WaitCounterImpl& getInstance(std::string_view key) {
    static auto& implMapSynchronized = *new Synchronized<
        std::unordered_map<std::string, std::unique_ptr<WaitCounterImpl>>>();

    return *implMapSynchronized.withLock([&](auto& implMap) {
      if (auto implIt = implMap.find(std::string(key));
          implIt != implMap.end()) {
        return implIt->second.get();
      }

      auto [implIt, emplaceSuccess] = implMap.emplace(
          std::string{key},
          std::unique_ptr<WaitCounterImpl>(new WaitCounterImpl(key)));

      assert(emplaceSuccess);

      return implIt->second.get();
    });
  }

  SmallVector<intptr_t> start() noexcept {
    auto now = std::chrono::steady_clock::now();
    SmallVector<intptr_t> ctxs;
    ctxs.reserve(backends_.size());
    for (const auto& backend : backends_) {
      ctxs.push_back(backend->start(now));
    }
    return ctxs;
  }

  void stop(const SmallVector<intptr_t>& ctxs) noexcept {
    auto now = std::chrono::steady_clock::now();
    assert(ctxs.size() == backends_.size());
    for (size_t i = 0; i < ctxs.size(); ++i) {
      backends_[i]->stop(now, ctxs[i]);
    }
  }

 private:
  explicit WaitCounterImpl(std::string_view key) {
    auto factoriesCopy = waitCounterBackendFactories().withLock(
        [](auto& factories) { return factories; });
    for (const auto& factory : factoriesCopy) {
      if (auto backend = factory->create(key)) {
        backends_.push_back(std::move(backend));
      }
    }
    if (auto backend = getDynamicBackend(key)) {
      backends_.push_back(std::move(backend));
    }
  }

  SmallVector<std::unique_ptr<WaitCounterBackendIf>> backends_;
};

void registerWaitCounterBackend(
    std::unique_ptr<WaitCounterBackendFactoryIf> factory) {
  waitCounterBackendFactories().withLock(
      [&](auto& factories) { factories.push_back(std::move(factory)); });
}

std::vector<std::shared_ptr<WaitCounterBackendFactoryIf>>
getRegisteredWaitCounterBackends() {
  return waitCounterBackendFactories().withLock(
      [](auto& factories) { return factories; });
}
} // namespace detail

WaitCounterHandle::WaitCounterHandle(std::string_view key)
    : impl_(detail::WaitCounterImpl::getInstance(key)) {}

WaitCounterHandle::WaitGuard WaitCounterHandle::start() {
  return WaitCounterHandle::WaitGuard(*this, impl_.start());
}

void WaitCounterHandle::stop(const SmallVector<intptr_t>& ctxs) {
  impl_.stop(ctxs);
}
} // namespace c10::monitor

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `class`, `detail`, `c10`

**Classes/Structs**: `DynamicBackendWrapper`, `WaitCounterImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/WaitCounter.h`
- `c10/util/Synchronized.h`
- `c10/util/WaitCounterDynamicBackend.h`
- `chrono`
- `memory`
- `string`
- `string_view`
- `unordered_map`
- `vector`
- `dlfcn.h`


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

- **File Documentation**: `WaitCounter.cpp_docs.md`
- **Keyword Index**: `WaitCounter.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
