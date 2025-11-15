# Documentation: `c10/util/OptionalArrayRef.h`

## File Metadata

- **Path**: `c10/util/OptionalArrayRef.h`
- **Size**: 7,184 bytes (7.02 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// This file defines OptionalArrayRef<T>, a class that has almost the same
// exact functionality as std::optional<ArrayRef<T>>, except that its
// converting constructor fixes a dangling pointer issue.
//
// The implicit converting constructor of both std::optional<ArrayRef<T>> and
// std::optional<ArrayRef<T>> can cause the underlying ArrayRef<T> to store
// a dangling pointer. OptionalArrayRef<T> prevents this by wrapping
// a std::optional<ArrayRef<T>> and fixing the constructor implementation.
//
// See https://github.com/pytorch/pytorch/issues/63645 for more on this.

#pragma once

#include <c10/util/ArrayRef.h>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <type_traits>
#include <utility>

namespace c10 {

template <typename T>
class OptionalArrayRef final {
 public:
  // Constructors

  constexpr OptionalArrayRef() noexcept = default;

  constexpr OptionalArrayRef(std::nullopt_t /*unused*/) noexcept {}

  OptionalArrayRef(const OptionalArrayRef& other) = default;

  OptionalArrayRef(OptionalArrayRef&& other) noexcept = default;

  constexpr OptionalArrayRef(const std::optional<ArrayRef<T>>& other) noexcept
      : wrapped_opt_array_ref(other) {}

  constexpr OptionalArrayRef(std::optional<ArrayRef<T>>&& other) noexcept
      : wrapped_opt_array_ref(std::move(other)) {}

  constexpr OptionalArrayRef(const T& value) noexcept
      : wrapped_opt_array_ref(value) {}

  template <
      typename U = ArrayRef<T>,
      std::enable_if_t<
          !std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
              !std::is_same_v<std::decay_t<U>, std::in_place_t> &&
              std::is_constructible_v<ArrayRef<T>, U&&> &&
              std::is_convertible_v<U&&, ArrayRef<T>> &&
              !std::is_convertible_v<U&&, T>,
          bool> = false>
  constexpr OptionalArrayRef(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&>)
      : wrapped_opt_array_ref(std::forward<U>(value)) {}

  template <
      typename U = ArrayRef<T>,
      std::enable_if_t<
          !std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
              !std::is_same_v<std::decay_t<U>, std::in_place_t> &&
              std::is_constructible_v<ArrayRef<T>, U&&> &&
              !std::is_convertible_v<U&&, ArrayRef<T>>,
          bool> = false>
  constexpr explicit OptionalArrayRef(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&>)
      : wrapped_opt_array_ref(std::forward<U>(value)) {}

  template <typename... Args>
  constexpr explicit OptionalArrayRef(
      std::in_place_t ip,
      Args&&... args) noexcept
      : wrapped_opt_array_ref(ip, std::forward<Args>(args)...) {}

  template <typename U, typename... Args>
  constexpr explicit OptionalArrayRef(
      std::in_place_t ip,
      std::initializer_list<U> il,
      Args&&... args)
      : wrapped_opt_array_ref(ip, il, std::forward<Args>(args)...) {}

  constexpr OptionalArrayRef(const std::initializer_list<T>& Vec)
      : wrapped_opt_array_ref(ArrayRef<T>(Vec)) {}

  // Destructor

  ~OptionalArrayRef() = default;

  // Assignment

  constexpr OptionalArrayRef& operator=(std::nullopt_t /*unused*/) noexcept {
    wrapped_opt_array_ref = std::nullopt;
    return *this;
  }

  OptionalArrayRef& operator=(const OptionalArrayRef& other) = default;

  OptionalArrayRef& operator=(OptionalArrayRef&& other) noexcept = default;

  constexpr OptionalArrayRef& operator=(
      const std::optional<ArrayRef<T>>& other) noexcept {
    wrapped_opt_array_ref = other;
    return *this;
  }

  constexpr OptionalArrayRef& operator=(
      std::optional<ArrayRef<T>>&& other) noexcept {
    wrapped_opt_array_ref = std::move(other);
    return *this;
  }

  template <
      typename U = ArrayRef<T>,
      typename = std::enable_if_t<
          !std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
          std::is_constructible_v<ArrayRef<T>, U&&> &&
          std::is_assignable_v<ArrayRef<T>&, U&&>>>
  constexpr OptionalArrayRef& operator=(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&> &&
      std::is_nothrow_assignable_v<ArrayRef<T>&, U&&>) {
    wrapped_opt_array_ref = std::forward<U>(value);
    return *this;
  }

  // Observers

  constexpr ArrayRef<T>* operator->() noexcept {
    return &wrapped_opt_array_ref.value();
  }

  constexpr const ArrayRef<T>* operator->() const noexcept {
    return &wrapped_opt_array_ref.value();
  }

  constexpr ArrayRef<T>& operator*() & noexcept {
    return wrapped_opt_array_ref.value();
  }

  constexpr const ArrayRef<T>& operator*() const& noexcept {
    return wrapped_opt_array_ref.value();
  }

  constexpr ArrayRef<T>&& operator*() && noexcept {
    return std::move(wrapped_opt_array_ref.value());
  }

  constexpr const ArrayRef<T>&& operator*() const&& noexcept {
    return std::move(wrapped_opt_array_ref.value());
  }

  constexpr explicit operator bool() const noexcept {
    return wrapped_opt_array_ref.has_value();
  }

  constexpr bool has_value() const noexcept {
    return wrapped_opt_array_ref.has_value();
  }

  constexpr ArrayRef<T>& value() & {
    return wrapped_opt_array_ref.value();
  }

  constexpr const ArrayRef<T>& value() const& {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return wrapped_opt_array_ref.value();
  }

  constexpr ArrayRef<T>&& value() && {
    return std::move(wrapped_opt_array_ref.value());
  }

  constexpr const ArrayRef<T>&& value() const&& {
    return std::move(wrapped_opt_array_ref.value());
  }

  template <typename U>
  constexpr std::
      enable_if_t<std::is_convertible_v<U&&, ArrayRef<T>>, ArrayRef<T>>
      value_or(U&& default_value) const& {
    return wrapped_opt_array_ref.value_or(std::forward<U>(default_value));
  }

  template <typename U>
  constexpr std::
      enable_if_t<std::is_convertible_v<U&&, ArrayRef<T>>, ArrayRef<T>>
      value_or(U&& default_value) && {
    return wrapped_opt_array_ref.value_or(std::forward<U>(default_value));
  }

  // Modifiers

  constexpr void swap(OptionalArrayRef& other) noexcept {
    std::swap(wrapped_opt_array_ref, other.wrapped_opt_array_ref);
  }

  constexpr void reset() noexcept {
    wrapped_opt_array_ref.reset();
  }

  template <typename... Args>
  constexpr std::
      enable_if_t<std::is_constructible_v<ArrayRef<T>, Args&&...>, ArrayRef<T>&>
      emplace(Args&&... args) noexcept(
          std::is_nothrow_constructible_v<ArrayRef<T>, Args&&...>) {
    return wrapped_opt_array_ref.emplace(std::forward<Args>(args)...);
  }

  template <typename U, typename... Args>
  constexpr ArrayRef<T>& emplace(
      std::initializer_list<U> il,
      Args&&... args) noexcept {
    return wrapped_opt_array_ref.emplace(il, std::forward<Args>(args)...);
  }

 private:
  std::optional<ArrayRef<T>> wrapped_opt_array_ref;
};

using OptionalIntArrayRef = OptionalArrayRef<int64_t>;

inline bool operator==(
    const OptionalIntArrayRef& a1,
    const IntArrayRef& other) {
  if (!a1.has_value()) {
    return false;
  }
  return a1.value() == other;
}

inline bool operator==(
    const c10::IntArrayRef& a1,
    const c10::OptionalIntArrayRef& a2) {
  return a2 == a1;
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `that`, `OptionalArrayRef`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/ArrayRef.h`
- `cstdint`
- `initializer_list`
- `optional`
- `type_traits`
- `utility`


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

- **File Documentation**: `OptionalArrayRef.h_docs.md`
- **Keyword Index**: `OptionalArrayRef.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
