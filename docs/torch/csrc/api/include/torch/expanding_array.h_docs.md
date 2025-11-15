# Documentation: `torch/csrc/api/include/torch/expanding_array.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/expanding_array.h`
- **Size**: 6,673 bytes (6.52 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <optional>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace torch {

/// A utility class that accepts either a container of `D`-many values, or a
/// single value, which is internally repeated `D` times. This is useful to
/// represent parameters that are multidimensional, but often equally sized in
/// all dimensions. For example, the kernel size of a 2D convolution has an `x`
/// and `y` length, but `x` and `y` are often equal. In such a case you could
/// just pass `3` to an `ExpandingArray<2>` and it would "expand" to `{3, 3}`.
template <size_t D, typename T = int64_t>
class ExpandingArray {
 public:
  /// Constructs an `ExpandingArray` from an `initializer_list`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(std::initializer_list<T> list)
      : ExpandingArray(c10::ArrayRef<T>(list)) {}

  /// Constructs an `ExpandingArray` from an `std::vector`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(std::vector<T> vec)
      : ExpandingArray(c10::ArrayRef<T>(vec)) {}

  /// Constructs an `ExpandingArray` from an `c10::ArrayRef`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(c10::ArrayRef<T> values) {
    // clang-format off
    TORCH_CHECK(
        values.size() == D,
        "Expected ", D, " values, but instead got ", values.size());
    // clang-format on
    std::copy(values.begin(), values.end(), values_.begin());
  }

  /// Constructs an `ExpandingArray` from a single value, which is repeated `D`
  /// times (where `D` is the extent parameter of the `ExpandingArray`).
  /*implicit*/ ExpandingArray(T single_size) {
    values_.fill(single_size);
  }

  /// Constructs an `ExpandingArray` from a correctly sized `std::array`.
  /*implicit*/ ExpandingArray(const std::array<T, D>& values)
      : values_(values) {}

  /// Accesses the underlying `std::array`.
  std::array<T, D>& operator*() {
    return values_;
  }

  /// Accesses the underlying `std::array`.
  const std::array<T, D>& operator*() const {
    return values_;
  }

  /// Accesses the underlying `std::array`.
  std::array<T, D>* operator->() {
    return &values_;
  }

  /// Accesses the underlying `std::array`.
  const std::array<T, D>* operator->() const {
    return &values_;
  }

  /// Returns an `ArrayRef` to the underlying `std::array`.
  operator c10::ArrayRef<T>() const {
    return values_;
  }

  /// Returns the extent of the `ExpandingArray`.
  size_t size() const noexcept {
    return D;
  }

 protected:
  /// The backing array.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::array<T, D> values_;
};

template <size_t D, typename T>
std::ostream& operator<<(
    std::ostream& stream,
    const ExpandingArray<D, T>& expanding_array) {
  if (expanding_array.size() == 1) {
    return stream << expanding_array->at(0);
  }
  return stream << static_cast<c10::ArrayRef<T>>(expanding_array);
}

/// A utility class that accepts either a container of `D`-many
/// `std::optional<T>` values, or a single `std::optional<T>` value, which is
/// internally repeated `D` times. It has the additional ability to accept
/// containers of the underlying type `T` and convert them to a container of
/// `std::optional<T>`.
template <size_t D, typename T = int64_t>
class ExpandingArrayWithOptionalElem
    : public ExpandingArray<D, std::optional<T>> {
 public:
  using ExpandingArray<D, std::optional<T>>::ExpandingArray;

  /// Constructs an `ExpandingArrayWithOptionalElem` from an `initializer_list`
  /// of the underlying type `T`. The extent of the length is checked against
  /// the `ExpandingArrayWithOptionalElem`'s extent parameter `D` at runtime.
  /*implicit*/ ExpandingArrayWithOptionalElem(std::initializer_list<T> list)
      : ExpandingArrayWithOptionalElem(c10::ArrayRef<T>(list)) {}

  /// Constructs an `ExpandingArrayWithOptionalElem` from an `std::vector` of
  /// the underlying type `T`. The extent of the length is checked against the
  /// `ExpandingArrayWithOptionalElem`'s extent parameter `D` at runtime.
  /*implicit*/ ExpandingArrayWithOptionalElem(std::vector<T> vec)
      : ExpandingArrayWithOptionalElem(c10::ArrayRef<T>(vec)) {}

  /// Constructs an `ExpandingArrayWithOptionalElem` from an `c10::ArrayRef` of
  /// the underlying type `T`. The extent of the length is checked against the
  /// `ExpandingArrayWithOptionalElem`'s extent parameter `D` at runtime.
  /*implicit*/ ExpandingArrayWithOptionalElem(c10::ArrayRef<T> values)
      : ExpandingArray<D, std::optional<T>>(0) {
    // clang-format off
    TORCH_CHECK(
        values.size() == D,
        "Expected ", D, " values, but instead got ", values.size());
    // clang-format on
    for (const auto i : c10::irange(this->values_.size())) {
      this->values_[i] = values[i];
    }
  }

  /// Constructs an `ExpandingArrayWithOptionalElem` from a single value of the
  /// underlying type `T`, which is repeated `D` times (where `D` is the extent
  /// parameter of the `ExpandingArrayWithOptionalElem`).
  /*implicit*/ ExpandingArrayWithOptionalElem(T single_size)
      : ExpandingArray<D, std::optional<T>>(0) {
    for (const auto i : c10::irange(this->values_.size())) {
      this->values_[i] = single_size;
    }
  }

  /// Constructs an `ExpandingArrayWithOptionalElem` from a correctly sized
  /// `std::array` of the underlying type `T`.
  /*implicit*/ ExpandingArrayWithOptionalElem(const std::array<T, D>& values)
      : ExpandingArray<D, std::optional<T>>(0) {
    for (const auto i : c10::irange(this->values_.size())) {
      this->values_[i] = values[i];
    }
  }
};

template <size_t D, typename T>
std::ostream& operator<<(
    std::ostream& stream,
    const ExpandingArrayWithOptionalElem<D, T>& expanding_array_with_opt_elem) {
  if (expanding_array_with_opt_elem.size() == 1) {
    const auto& elem = expanding_array_with_opt_elem->at(0);
    stream << (elem.has_value() ? c10::str(elem.value()) : "None");
  } else {
    std::vector<std::string> str_array;
    for (const auto& elem : *expanding_array_with_opt_elem) {
      str_array.emplace_back(
          elem.has_value() ? c10::str(elem.value()) : "None");
    }
    stream << c10::ArrayRef<std::string>(str_array);
  }
  return stream;
}

} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `that`, `ExpandingArray`, `that`, `ExpandingArrayWithOptionalElem`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/ArrayRef.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`
- `optional`
- `algorithm`
- `array`
- `cstdint`
- `initializer_list`
- `string`
- `vector`


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

Files in the same folder (`torch/csrc/api/include/torch`):

- [`ordered_dict.h_docs.md`](./ordered_dict.h_docs.md)
- [`fft.h_docs.md`](./fft.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`serialize.h_docs.md`](./serialize.h_docs.md)
- [`nn.h_docs.md`](./nn.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`special.h_docs.md`](./special.h_docs.md)
- [`data.h_docs.md`](./data.h_docs.md)
- [`version.h_docs.md`](./version.h_docs.md)


## Cross-References

- **File Documentation**: `expanding_array.h_docs.md`
- **Keyword Index**: `expanding_array.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
