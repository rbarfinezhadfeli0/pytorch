# Documentation: `torch/csrc/distributed/c10d/Store.hpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/Store.hpp`
- **Size**: 4,304 bytes (4.20 KB)
- **Type**: C++ Header File
- **Extension**: `.hpp`

## File Purpose

This is a c++ header file that is part of the PyTorch project.

## Original Source

```cpp
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <c10/macros/Macros.h>
#include <torch/custom_class.h>

namespace c10d {

// callback function will be given arguments (std::optional<string> oldValue,
// std::optional<string> newValue)
using WatchKeyCallback =
    std::function<void(std::optional<std::string>, std::optional<std::string>)>;

class TORCH_API Store : public torch::CustomClassHolder {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(300);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();

  Store() : timeout_(kDefaultTimeout) {}

  explicit Store(const std::chrono::milliseconds& timeout)
      : timeout_(timeout) {}

  Store(const Store&) = default;
  Store(Store&&) noexcept = default;

  ~Store() override = default;

  // Clone a thread safe copy of this store object that points to the same
  // underlying store.
  virtual c10::intrusive_ptr<Store> clone() = 0;

  void set(const std::string& key, const std::string& value);

  virtual void set(
      const std::string& key,
      const std::vector<uint8_t>& value) = 0;

  std::string compareSet(
      const std::string& key,
      const std::string& currentValue,
      const std::string& newValue);

  virtual std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& currentValue,
      const std::vector<uint8_t>& newValue) {
    C10_THROW_ERROR(NotImplementedError, "Not implemented.");
  }

  std::string get_to_str(const std::string& key);

  virtual std::vector<uint8_t> get(const std::string& key) = 0;

  virtual int64_t add(const std::string& key, int64_t value) = 0;

  virtual bool deleteKey(const std::string& key) = 0;

  virtual bool check(const std::vector<std::string>& keys) = 0;

  virtual int64_t getNumKeys() = 0;

  virtual void wait(const std::vector<std::string>& keys) = 0;

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) = 0;

  virtual const std::chrono::milliseconds& getTimeout() const noexcept;

  virtual void setTimeout(const std::chrono::milliseconds& timeout);

  // watchKey() is deprecated and no longer supported.
  virtual void watchKey(
      const std::string& /* unused */,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      WatchKeyCallback /* unused */) {
    C10_THROW_ERROR(
        NotImplementedError,
        "watchKey is deprecated, no implementation support it.");
  }

  virtual void append(
      const std::string& key,
      const std::vector<uint8_t>& value);

  virtual std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys);

  virtual void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values);

  // Returns true if this store support append, multiGet and multiSet
  virtual bool hasExtendedApi() const;

  virtual void queuePush(
      const std::string& key,
      const std::vector<uint8_t>& value) {
    C10_THROW_ERROR(NotImplementedError, "queue support is not implemented.");
  }

  virtual std::vector<uint8_t> queuePop(const std::string& key, bool block) {
    C10_THROW_ERROR(NotImplementedError, "queue support is not implemented.");
  }

  virtual int64_t queueLen(const std::string& key) {
    C10_THROW_ERROR(NotImplementedError, "queue support is not implemented.");
  }

 protected:
  std::chrono::milliseconds timeout_;
};

/*
StoreTimeoutGuard is a RAII guard that will set the store timeout and restore it
when it returns.
*/
class StoreTimeoutGuard {
 public:
  explicit StoreTimeoutGuard(
      Store& store,
      const std::chrono::milliseconds& timeout)
      : store_(store), oldTimeout_(store.getTimeout()) {
    store.setTimeout(timeout);
  }

  ~StoreTimeoutGuard() {
    store_.setTimeout(oldTimeout_);
  }

  /* Disabling copy and move semantics */
  StoreTimeoutGuard(const StoreTimeoutGuard&) = delete;
  StoreTimeoutGuard& operator=(const StoreTimeoutGuard&) = delete;
  StoreTimeoutGuard(StoreTimeoutGuard&&) = delete;
  StoreTimeoutGuard& operator=(StoreTimeoutGuard&&) = delete;

 private:
  Store& store_;
  std::chrono::milliseconds oldTimeout_{};
};

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `TORCH_API`, `StoreTimeoutGuard`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `chrono`
- `cstdint`
- `string`
- `vector`
- `c10/macros/Macros.h`
- `torch/custom_class.h`


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

Files in the same folder (`torch/csrc/distributed/c10d`):

- [`Utils.hpp_docs.md`](./Utils.hpp_docs.md)
- [`Ops.cpp_docs.md`](./Ops.cpp_docs.md)
- [`WinSockUtils.hpp_docs.md`](./WinSockUtils.hpp_docs.md)
- [`FakeProcessGroup.hpp_docs.md`](./FakeProcessGroup.hpp_docs.md)
- [`Work.cpp_docs.md`](./Work.cpp_docs.md)
- [`PrefixStore.hpp_docs.md`](./PrefixStore.hpp_docs.md)
- [`PyProcessGroup.hpp_docs.md`](./PyProcessGroup.hpp_docs.md)
- [`debug.h_docs.md`](./debug.h_docs.md)
- [`exception.h_docs.md`](./exception.h_docs.md)


## Cross-References

- **File Documentation**: `Store.hpp_docs.md`
- **Keyword Index**: `Store.hpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
