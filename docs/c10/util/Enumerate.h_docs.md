# Documentation: `c10/util/Enumerate.h`

## File Metadata

- **Path**: `c10/util/Enumerate.h`
- **Size**: 4,001 bytes (3.91 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*
 * Ported from folly/container/Enumerate.h
 */

#pragma once

#include <iterator>
#include <memory>

#ifdef _WIN32
#include <basetsd.h> // @manual
using ssize_t = SSIZE_T;
#endif

#include <c10/macros/Macros.h>

/**
 * Similar to Python's enumerate(), enumerate() can be used to
 * iterate a range with a for-range loop, and it also allows to
 * retrieve the count of iterations so far. Can be used in constexpr
 * context.
 *
 * For example:
 *
 * for (auto&& [index, element] : enumerate(vec)) {
 *   // index is a const reference to a size_t containing the iteration count.
 *   // element is a reference to the type contained within vec, mutable
 *   // unless vec is const.
 * }
 *
 * If the binding is const, the element reference is too.
 *
 * for (const auto&& [index, element] : enumerate(vec)) {
 *   // element is always a const reference.
 * }
 *
 * It can also be used as follows:
 *
 * for (auto&& it : enumerate(vec)) {
 *   // *it is a reference to the current element. Mutable unless vec is const.
 *   // it->member can be used as well.
 *   // it.index contains the iteration count.
 * }
 *
 * As before, const auto&& it can also be used.
 */

namespace c10 {

namespace detail {

template <class T>
struct MakeConst {
  using type = const T;
};
template <class T>
struct MakeConst<T&> {
  using type = const T&;
};
template <class T>
struct MakeConst<T*> {
  using type = const T*;
};

template <class Iterator>
class Enumerator {
 public:
  constexpr explicit Enumerator(Iterator it) : it_(std::move(it)) {}

  class Proxy {
   public:
    using difference_type = ssize_t;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference = typename std::iterator_traits<Iterator>::reference;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using iterator_category = std::input_iterator_tag;

    C10_ALWAYS_INLINE constexpr explicit Proxy(const Enumerator& e)
        : index(e.idx_), element(*e.it_) {}

    // Non-const Proxy: Forward constness from Iterator.
    C10_ALWAYS_INLINE constexpr reference operator*() {
      return element;
    }
    C10_ALWAYS_INLINE constexpr pointer operator->() {
      return std::addressof(element);
    }

    // Const Proxy: Force const references.
    C10_ALWAYS_INLINE constexpr typename MakeConst<reference>::type operator*()
        const {
      return element;
    }
    C10_ALWAYS_INLINE constexpr typename MakeConst<pointer>::type operator->()
        const {
      return std::addressof(element);
    }

   public:
    size_t index;
    reference element;
  };

  C10_ALWAYS_INLINE constexpr Proxy operator*() const {
    return Proxy(*this);
  }

  C10_ALWAYS_INLINE constexpr Enumerator& operator++() {
    ++it_;
    ++idx_;
    return *this;
  }

  template <typename OtherIterator>
  C10_ALWAYS_INLINE constexpr bool operator==(
      const Enumerator<OtherIterator>& rhs) const {
    return it_ == rhs.it_;
  }

  template <typename OtherIterator>
  C10_ALWAYS_INLINE constexpr bool operator!=(
      const Enumerator<OtherIterator>& rhs) const {
    return !(it_ == rhs.it_);
  }

 private:
  template <typename OtherIterator>
  friend class Enumerator;

  Iterator it_;
  size_t idx_ = 0;
};

template <class Range>
class RangeEnumerator {
  Range r_;
  using BeginIteratorType = decltype(std::declval<Range>().begin());
  using EndIteratorType = decltype(std::declval<Range>().end());

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  constexpr explicit RangeEnumerator(Range&& r) : r_(std::forward<Range>(r)) {}

  constexpr Enumerator<BeginIteratorType> begin() {
    return Enumerator<BeginIteratorType>(r_.begin());
  }
  constexpr Enumerator<EndIteratorType> end() {
    return Enumerator<EndIteratorType>(r_.end());
  }
};

} // namespace detail

template <class Range>
constexpr detail::RangeEnumerator<Range> enumerate(Range&& r) {
  return detail::RangeEnumerator<Range>(std::forward<Range>(r));
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 10 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `T`, `MakeConst`, `T`, `MakeConst`, `T`, `MakeConst`, `Iterator`, `Enumerator`, `Proxy`, `Enumerator`, `Range`, `RangeEnumerator`, `Range`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `iterator`
- `memory`
- `basetsd.h`
- `c10/macros/Macros.h`


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

- **File Documentation**: `Enumerate.h_docs.md`
- **Keyword Index**: `Enumerate.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
