# Documentation: any_value.h

## File Metadata
- **Path**: `torch/csrc/api/include/torch/nn/modules/container/any_value.h`
- **Size**: 4138 bytes
- **Lines**: 124
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
#pragma once

#include <torch/types.h>

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace torch::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyValue ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// An implementation of `std::any` which stores
/// a type erased object, whose concrete value can be retrieved at runtime by
/// checking if the `typeid()` of a requested type matches the `typeid()` of
/// the object stored.
class AnyValue {
 public:
  /// Move construction and assignment is allowed, and follows the default
  /// behavior of move for `std::unique_ptr`.
  AnyValue(AnyValue&&) = default;
  AnyValue& operator=(AnyValue&&) = default;
  ~AnyValue() = default;

  /// Copy construction and assignment is allowed.
  AnyValue(const AnyValue& other) : content_(other.content_->clone()) {}
  AnyValue& operator=(const AnyValue& other) {
    content_ = other.content_->clone();
    return *this;
  }

  /// Constructs the `AnyValue` from value type.
  template <
      typename T,
      typename = std::enable_if_t<!std::is_same_v<T, AnyValue>>>
  explicit AnyValue(T&& value)
      : content_(
            std::make_unique<Holder<std::decay_t<T>>>(std::forward<T>(value))) {
  }

  /// Returns a pointer to the value contained in the `AnyValue` if the type
  /// passed as template parameter matches the type of the value stored, and
  /// returns a null pointer otherwise.
  template <typename T>
  T* try_get() {
    static_assert(
        !std::is_reference_v<T>,
        "AnyValue stores decayed types, you cannot cast it to a reference type");
    static_assert(
        !std::is_array_v<T>,
        "AnyValue stores decayed types, you must cast it to T* instead of T[]");
    if (typeid(T).hash_code() == type_info().hash_code()) {
      return &static_cast<Holder<T>&>(*content_).value;
    }
    return nullptr;
  }

  /// Returns the value contained in the `AnyValue` if the type passed as
  /// template parameter matches the type of the value stored, and throws an
  /// exception otherwise.
  template <typename T>
  T get() {
    if (auto* maybe_value = try_get<T>()) {
      return *maybe_value;
    }
    TORCH_CHECK(
        false,
        "Attempted to cast AnyValue to ",
        c10::demangle(typeid(T).name()),
        ", but its actual type is ",
        c10::demangle(type_info().name()));
  }

  /// Returns the `type_info` object of the contained value.
  const std::type_info& type_info() const noexcept {
    return content_->type_info;
  }

 private:
  friend struct AnyModulePlaceholder;
  friend struct TestAnyValue;

  /// \internal
  /// The static type of the object we store in the `AnyValue`, which erases the
  /// actual object's type, allowing us only to check the `type_info` of the
  /// type stored in the dynamic type.
  struct Placeholder {
    explicit Placeholder(const std::type_info& type_info_) noexcept
        : type_info(type_info_) {}
    Placeholder(const Placeholder&) = default;
    Placeholder(Placeholder&&) = default;
    Placeholder& operator=(const Placeholder&) = delete;
    Placeholder& operator=(Placeholder&&) = delete;
    virtual ~Placeholder() = default;
    virtual std::unique_ptr<Placeholder> clone() const {
      TORCH_CHECK(false, "clone() should only be called on `AnyValue::Holder`");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::type_info& type_info;
  };

  /// \internal
  /// The dynamic type of the object we store in the `AnyValue`, which hides the
  /// actual object we have erased in this `AnyValue`.
  template <typename T>
  struct Holder : public Placeholder {
    /// A template because T&& would not be universal reference here.
    template <
        typename U,
        typename = std::enable_if_t<!std::is_same_v<U, Holder>>>
    explicit Holder(U&& value_) noexcept
        : Placeholder(typeid(T)), value(std::forward<U>(value_)) {}
    std::unique_ptr<Placeholder> clone() const override {
      return std::make_unique<Holder<T>>(value);
    }
    T value;
  };

  /// The type erased object.
  std::unique_ptr<Placeholder> content_;
};

} // namespace torch::nn

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): AnyValue

### Structures
This file defines 4 struct(s): AnyModulePlaceholder, TestAnyValue, Placeholder, Holder


## Key Components

The file contains 474 words across 124 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4138 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
