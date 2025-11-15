# Documentation: `torch/csrc/lazy/core/util.h`

## File Metadata

- **Path**: `torch/csrc/lazy/core/util.h`
- **Size**: 2,873 bytes (2.81 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/**
 * Most of the utils in this file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/util.h
 */

#pragma once

#include <exception>
#include <functional>
#include <vector>

#include <c10/util/OptionalArrayRef.h>
#include <optional>

namespace torch::lazy {

// Similar to c10::scope_exit but with a status.
// TODO(alanwaketan): Consolidate it with c10::scope_exit.
template <typename T>
class Cleanup {
 public:
  using StatusType = T;

  explicit Cleanup(std::function<void(StatusType&&)>&& func)
      : func_(std::move(func)) {}
  Cleanup(Cleanup&& ref) noexcept
      : func_(std::move(ref.func_)), status_(std::move(ref.status_)) {}
  Cleanup(const Cleanup&) = delete;

  ~Cleanup() {
    if (func_ != nullptr) {
      func_(std::move(status_));
    }
  }

  Cleanup& operator=(const Cleanup&) = delete;

  Cleanup& operator=(Cleanup&& ref) noexcept {
    if (this != &ref) {
      func_ = std::move(ref.func_);
      status_ = std::move(ref.status_);
    }
    return *this;
  }

  void Release() {
    func_ = nullptr;
  }

  void SetStatus(StatusType&& status) {
    status_ = std::move(status);
  }

  const StatusType& GetStatus() const {
    return status_;
  }

 private:
  std::function<void(StatusType&&)> func_;
  StatusType status_;
};

using ExceptionCleanup = Cleanup<std::exception_ptr>;

// Allows APIs which might return const references and values, to not be forced
// to return values in the signature.
// TODO(alanwaketan): This is clever, but is there really no std or c10
// supports? Needs more investigations.
template <typename T>
class MaybeRef {
 public:
  /* implicit */ MaybeRef(const T& ref) : ref_(ref) {}
  /* implicit */ MaybeRef(T&& value)
      : storage_(std::move(value)), ref_(*storage_) {}

  const T& Get() const {
    return ref_;
  }
  const T& operator*() const {
    return Get();
  }
  operator const T&() const {
    return Get();
  }

  bool IsStored() const {
    return storage_.has_value();
  }

 private:
  std::optional<T> storage_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const T& ref_;
};

template <typename T>
std::vector<T> Iota(size_t size, T init = 0, T incr = 1) {
  std::vector<T> result(size);
  T value = init;
  for (size_t i = 0; i < size; ++i, value += incr) {
    result[i] = value;
  }
  return result;
}

template <typename T, typename S>
std::vector<T> ToVector(const S& input) {
  return std::vector<T>(input.begin(), input.end());
}

template <typename T>
std::optional<std::vector<T>> ToOptionalVector(
    c10::OptionalArrayRef<T> arrayRef) {
  if (arrayRef) {
    return arrayRef->vec();
  }
  return std::nullopt;
}

template <typename T>
std::underlying_type_t<T> GetEnumValue(T value) {
  return static_cast<std::underlying_type_t<T>>(value);
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Cleanup`, `MaybeRef`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `exception`
- `functional`
- `vector`
- `c10/util/OptionalArrayRef.h`
- `optional`


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

Files in the same folder (`torch/csrc/lazy/core`):

- [`hash.cpp_docs.md`](./hash.cpp_docs.md)
- [`shape_inference.cpp_docs.md`](./shape_inference.cpp_docs.md)
- [`tensor_impl.h_docs.md`](./tensor_impl.h_docs.md)
- [`helpers.h_docs.md`](./helpers.h_docs.md)
- [`tensor_impl.cpp_docs.md`](./tensor_impl.cpp_docs.md)
- [`ir_metadata.cpp_docs.md`](./ir_metadata.cpp_docs.md)
- [`ir_metadata.h_docs.md`](./ir_metadata.h_docs.md)
- [`trie.cpp_docs.md`](./trie.cpp_docs.md)
- [`cache.h_docs.md`](./cache.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)


## Cross-References

- **File Documentation**: `util.h_docs.md`
- **Keyword Index**: `util.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
