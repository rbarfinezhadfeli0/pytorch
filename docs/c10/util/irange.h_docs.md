# Documentation: `c10/util/irange.h`

## File Metadata

- **Path**: `c10/util/irange.h`
- **Size**: 3,436 bytes (3.36 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <c10/util/TypeSafeSignMath.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace c10 {

namespace detail {

template <
    typename I,
    bool one_sided = false,
    std::enable_if_t<std::is_integral_v<I>, int> = 0>
struct integer_iterator {
  using iterator_category = std::input_iterator_tag;
  using value_type = I;
  using difference_type = std::ptrdiff_t;
  using pointer = I*;
  using reference = I&;

  explicit constexpr integer_iterator(I val) : value(val) {}

  constexpr I operator*() const {
    return value;
  }

  constexpr I const* operator->() const {
    return &value;
  }

  constexpr integer_iterator& operator++() {
    ++value;
    return *this;
  }

  constexpr integer_iterator operator++(int) {
    const auto copy = *this;
    ++*this;
    return copy;
  }

  constexpr bool operator==(const integer_iterator& other) const {
    if constexpr (one_sided) {
      // Range-for loops' end test is `begin != end`, not `begin <
      // end`. To handle `c10::irange(n)` where n < 0 (which should be
      // empty), we just make `begin != end` fail whenever `end` is
      // negative.
      return is_negative(other.value) || value == other.value;
    } else {
      return value == other.value;
    }
    // Suppress "warning: missing return statement at end of non-void function"
    // which Nvidia's Robert Crovella confirms is an NVCC compiler error
    // here https://stackoverflow.com/a/64561686/752843 on 2020-10-27
    // `__builtin_unreachable();` would be best here, but it's not
    // available with all compilers. So we instead return an arbitrary
    // value trusting that this line will, in fact, never be reached.
    return false; // Horrible hack
  }

  constexpr bool operator!=(const integer_iterator& other) const {
    return !(*this == other);
  }

 protected:
  I value;
};

} // namespace detail

template <
    typename I,
    bool one_sided = false,
    std::enable_if_t<std::is_integral_v<I>, bool> = true>
struct integer_range {
 public:
  constexpr integer_range(I begin, I end) : begin_(begin), end_(end) {}
  using iterator = detail::integer_iterator<I, one_sided>;
  constexpr iterator begin() const {
    return begin_;
  }
  constexpr iterator end() const {
    return end_;
  }

 private:
  iterator begin_;
  iterator end_;
};

/// Creates an integer range for the half-open interval [begin, end)
/// If end<=begin, then the range is empty.
/// The range has the type of the `end` integer; `begin` integer is
/// cast to this type.
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<std::is_integral_v<Integer1>, bool> = true,
    std::enable_if_t<std::is_integral_v<Integer2>, bool> = true>
constexpr integer_range<Integer2> irange(Integer1 begin, Integer2 end) {
  // If end<=begin then the range is empty; we can achieve this effect by
  // choosing the larger of {begin, end} as the loop terminator
  return {
      static_cast<Integer2>(begin),
      std::max(static_cast<Integer2>(begin), end)};
}

/// Creates an integer range for the half-open interval [0, end)
/// If end<=begin, then the range is empty
template <
    typename Integer,
    std::enable_if_t<std::is_integral_v<Integer>, bool> = true>
constexpr integer_range<Integer, true> irange(Integer end) {
  return {Integer(), end};
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `integer_iterator`, `integer_range`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/TypeSafeSignMath.h`
- `algorithm`
- `cstddef`
- `iterator`
- `type_traits`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `irange.h_docs.md`
- **Keyword Index**: `irange.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
