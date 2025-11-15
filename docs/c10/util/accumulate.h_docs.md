# Documentation: `c10/util/accumulate.h`

## File Metadata

- **Path**: `c10/util/accumulate.h`
- **Size**: 4,032 bytes (3.94 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <c10/util/Exception.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>

namespace c10 {

/// Sum of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t sum_integers(const C& container) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      container.begin(), container.end(), static_cast<int64_t>(0));
}

/// Sum of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    std::enable_if_t<
        std::is_integral_v<typename std::iterator_traits<Iter>::value_type>,
        int> = 0>
inline int64_t sum_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(begin, end, static_cast<int64_t>(0));
}

/// Product of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t multiply_integers(const C& container) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      container.begin(),
      container.end(),
      static_cast<int64_t>(1),
      std::multiplies<>());
}

/// Product of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    std::enable_if_t<
        std::is_integral_v<typename std::iterator_traits<Iter>::value_type>,
        int> = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      begin, end, static_cast<int64_t>(1), std::multiplies<>());
}

/// Return product of all dimensions starting from k
/// Returns 1 if k>=dims.size()
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_from_dim(const int k, const C& dims) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(k >= 0);

  if (k > static_cast<int>(dims.size())) {
    return 1;
  } else {
    auto cbegin = dims.cbegin();
    std::advance(cbegin, k);
    return multiply_integers(cbegin, dims.cend());
  }
}

/// Product of all dims up to k (not including dims[k])
/// Throws an error if k>dims.size()
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_to_dim(const int k, const C& dims) {
  TORCH_INTERNAL_ASSERT(0 <= k);
  TORCH_INTERNAL_ASSERT((unsigned)k <= dims.size());

  auto cend = dims.cbegin();
  std::advance(cend, k);
  return multiply_integers(dims.cbegin(), cend);
}

/// Product of all dims between k and l (including dims[k] and excluding
/// dims[l]) k and l may be supplied in either order
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_between_dim(int k, int l, const C& dims) {
  TORCH_INTERNAL_ASSERT(0 <= k);
  TORCH_INTERNAL_ASSERT(0 <= l);

  if (k > l) {
    std::swap(k, l);
  }

  TORCH_INTERNAL_ASSERT((unsigned)l < dims.size());

  auto cbegin = dims.cbegin();
  auto cend = dims.cbegin();
  std::advance(cbegin, k);
  std::advance(cend, l);
  return multiply_integers(cbegin, cend);
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `cstdint`
- `functional`
- `iterator`
- `numeric`
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

- **File Documentation**: `accumulate.h_docs.md`
- **Keyword Index**: `accumulate.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
