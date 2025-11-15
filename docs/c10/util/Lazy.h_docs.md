# Documentation: `c10/util/Lazy.h`

## File Metadata

- **Path**: `c10/util/Lazy.h`
- **Size**: 2,823 bytes (2.76 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <atomic>
#include <utility>

namespace c10 {

/**
 * Thread-safe lazy value with opportunistic concurrency: on concurrent first
 * access, the factory may be called by multiple threads, but only one result is
 * stored and its reference returned to all the callers.
 *
 * Value is heap-allocated; this optimizes for the case in which the value is
 * never actually computed.
 */
template <class T>
class OptimisticLazy {
 public:
  OptimisticLazy() = default;
  OptimisticLazy(const OptimisticLazy& other) {
    if (T* value = other.value_.load(std::memory_order_acquire)) {
      value_ = new T(*value);
    }
  }
  OptimisticLazy(OptimisticLazy&& other) noexcept
      : value_(other.value_.exchange(nullptr, std::memory_order_acq_rel)) {}
  ~OptimisticLazy() {
    reset();
  }

  template <class Factory>
  T& ensure(const Factory& factory) {
    if (T* value = value_.load(std::memory_order_acquire)) {
      return *value;
    }
    T* value = new T(factory());
    T* old = nullptr;
    if (!value_.compare_exchange_strong(
            old, value, std::memory_order_release, std::memory_order_acquire)) {
      delete value;
      value = old;
    }
    return *value;
  }

  // The following methods are not thread-safe: they should not be called
  // concurrently with any other method.

  OptimisticLazy& operator=(const OptimisticLazy& other) {
    *this = OptimisticLazy{other};
    return *this;
  }

  OptimisticLazy& operator=(OptimisticLazy&& other) noexcept {
    if (this != &other) {
      reset();
      value_.store(
          other.value_.exchange(nullptr, std::memory_order_acquire),
          std::memory_order_release);
    }
    return *this;
  }

  void reset() {
    if (T* old = value_.load(std::memory_order_relaxed)) {
      value_.store(nullptr, std::memory_order_relaxed);
      delete old;
    }
  }

 private:
  std::atomic<T*> value_{nullptr};
};

/**
 * Interface for a value that is computed on first access.
 */
template <class T>
class LazyValue {
 public:
  virtual ~LazyValue() = default;

  virtual const T& get() const = 0;
};

/**
 * Convenience thread-safe LazyValue implementation with opportunistic
 * concurrency.
 */
template <class T>
class OptimisticLazyValue : public LazyValue<T> {
 public:
  const T& get() const override {
    return value_.ensure([this] { return compute(); });
  }

 private:
  virtual T compute() const = 0;

  mutable OptimisticLazy<T> value_;
};

/**
 * Convenience immutable (thus thread-safe) LazyValue implementation for cases
 * in which the value is not actually lazy.
 */
template <class T>
class PrecomputedLazyValue : public LazyValue<T> {
 public:
  PrecomputedLazyValue(T value) : value_(std::move(value)) {}

  const T& get() const override {
    return value_;
  }

 private:
  T value_;
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 9 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `T`, `OptimisticLazy`, `Factory`, `T`, `LazyValue`, `T`, `OptimisticLazyValue`, `T`, `PrecomputedLazyValue`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `atomic`
- `utility`


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

- **File Documentation**: `Lazy.h_docs.md`
- **Keyword Index**: `Lazy.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
