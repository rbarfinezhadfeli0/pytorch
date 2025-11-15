# Documentation: `torch/headeronly/util/TypeSafeSignMath.h`

## File Metadata

- **Path**: `torch/headeronly/util/TypeSafeSignMath.h`
- **Size**: 4,585 bytes (4.48 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <limits>
#include <type_traits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wstring-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wstring-conversion")
#endif
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

/// Returns false since we cannot have x < 0 if x is unsigned.
template <typename T>
inline constexpr bool is_negative(
    const T& /*x*/,
    std::true_type /*is_unsigned*/) {
  return false;
}

/// Returns true if a signed variable x < 0
template <typename T>
inline constexpr bool is_negative(const T& x, std::false_type /*is_unsigned*/) {
  return x < T(0);
}

/// Returns true if x < 0
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :-(
template <typename T>
inline constexpr bool is_negative(const T& x) {
  return is_negative(x, std::is_unsigned<T>());
}

/// Returns the sign of an unsigned variable x as 0, 1
template <typename T>
inline constexpr int signum(const T& x, std::true_type /*is_unsigned*/) {
  return T(0) < x;
}

/// Returns the sign of a signed variable x as -1, 0, 1
template <typename T>
inline constexpr int signum(const T& x, std::false_type /*is_unsigned*/) {
  return (T(0) < x) - (x < T(0));
}

/// Returns the sign of x as -1, 0, 1
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :-(
template <typename T>
inline constexpr int signum(const T& x) {
  return signum(x, std::is_unsigned<T>());
}

/// Returns true if a and b are not both negative
template <typename T, typename U>
inline constexpr bool signs_differ(const T& a, const U& b) {
  return is_negative(a) != is_negative(b);
}

// Suppress sign compare warning when compiling with GCC
// as later does not account for short-circuit rule before
// raising the warning, see https://godbolt.org/z/Tr3Msnz99
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

/// Returns true if x is greater than the greatest value of the type Limit
template <typename Limit, typename T>
inline constexpr bool greater_than_max(const T& x) {
  constexpr bool can_overflow =
      std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;
  return can_overflow && x > std::numeric_limits<Limit>::max();
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/// Returns true if x < lowest(Limit). Standard comparison
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(
    const T& x,
    std::false_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  return x < std::numeric_limits<Limit>::lowest();
}

/// Returns false since all the limit is signed and therefore includes
/// negative values but x cannot be negative because it is unsigned
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(
    const T& /*x*/,
    std::false_type /*limit_is_unsigned*/,
    std::true_type /*x_is_unsigned*/) {
  return false;
}

/// Returns true if x < 0, where 0 is constructed from T.
/// Limit is not signed, so its lower value is zero
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(
    const T& x,
    std::true_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  return x < T(0);
}

/// Returns false sign both types are unsigned
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(
    const T& /*x*/,
    std::true_type /*limit_is_unsigned*/,
    std::true_type /*x_is_unsigned*/) {
  return false;
}

/// Returns true if x is less than the lowest value of type T
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(const T& x) {
  return less_than_lowest<Limit>(
      x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::greater_than_max;
using c10::is_negative;
using c10::less_than_lowest;
using c10::signs_differ;
using c10::signum;
HIDDEN_NAMESPACE_END(torch, headeronly)

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/headeronly/util`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/headeronly/macros/Macros.h`
- `limits`
- `type_traits`


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

Files in the same folder (`torch/headeronly/util`):

- [`Float8_e4m3fnuz.h_docs.md`](./Float8_e4m3fnuz.h_docs.md)
- [`HeaderOnlyArrayRef.h_docs.md`](./HeaderOnlyArrayRef.h_docs.md)
- [`Float8_e5m2.h_docs.md`](./Float8_e5m2.h_docs.md)
- [`floating_point_utils.h_docs.md`](./floating_point_utils.h_docs.md)
- [`shim_utils.h_docs.md`](./shim_utils.h_docs.md)
- [`TypeList.h_docs.md`](./TypeList.h_docs.md)
- [`Float4_e2m1fn_x2.h_docs.md`](./Float4_e2m1fn_x2.h_docs.md)
- [`Float8_e8m0fnu.h_docs.md`](./Float8_e8m0fnu.h_docs.md)
- [`BFloat16.h_docs.md`](./BFloat16.h_docs.md)
- [`Float8_fnuz_cvt.h_docs.md`](./Float8_fnuz_cvt.h_docs.md)


## Cross-References

- **File Documentation**: `TypeSafeSignMath.h_docs.md`
- **Keyword Index**: `TypeSafeSignMath.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
