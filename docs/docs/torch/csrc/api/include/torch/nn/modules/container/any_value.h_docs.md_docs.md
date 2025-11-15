# Documentation: `docs/torch/csrc/api/include/torch/nn/modules/container/any_value.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/modules/container/any_value.h_docs.md`
- **Size**: 6,597 bytes (6.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/modules/container/any_value.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/container/any_value.h`
- **Size**: 4,138 bytes (4.04 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
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


This C++ file contains approximately 1 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `AnyValue`, `AnyModulePlaceholder`, `TestAnyValue`, `Placeholder`, `Holder`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules/container`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/types.h`
- `memory`
- `type_traits`
- `typeinfo`
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

Files in the same folder (`torch/csrc/api/include/torch/nn/modules/container`):

- [`moduledict.h_docs.md`](./moduledict.h_docs.md)
- [`sequential.h_docs.md`](./sequential.h_docs.md)
- [`any_module_holder.h_docs.md`](./any_module_holder.h_docs.md)
- [`parameterlist.h_docs.md`](./parameterlist.h_docs.md)
- [`modulelist.h_docs.md`](./modulelist.h_docs.md)
- [`named_any.h_docs.md`](./named_any.h_docs.md)
- [`functional.h_docs.md`](./functional.h_docs.md)
- [`any.h_docs.md`](./any.h_docs.md)
- [`parameterdict.h_docs.md`](./parameterdict.h_docs.md)


## Cross-References

- **File Documentation**: `any_value.h_docs.md`
- **Keyword Index**: `any_value.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn/modules/container`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn/modules/container`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/csrc/api/include/torch/nn/modules/container`):

- [`sequential.h_docs.md_docs.md`](./sequential.h_docs.md_docs.md)
- [`functional.h_docs.md_docs.md`](./functional.h_docs.md_docs.md)
- [`modulelist.h_docs.md_docs.md`](./modulelist.h_docs.md_docs.md)
- [`any.h_docs.md_docs.md`](./any.h_docs.md_docs.md)
- [`named_any.h_docs.md_docs.md`](./named_any.h_docs.md_docs.md)
- [`moduledict.h_docs.md_docs.md`](./moduledict.h_docs.md_docs.md)
- [`modulelist.h_kw.md_docs.md`](./modulelist.h_kw.md_docs.md)
- [`parameterdict.h_docs.md_docs.md`](./parameterdict.h_docs.md_docs.md)
- [`parameterlist.h_kw.md_docs.md`](./parameterlist.h_kw.md_docs.md)
- [`named_any.h_kw.md_docs.md`](./named_any.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `any_value.h_docs.md_docs.md`
- **Keyword Index**: `any_value.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
