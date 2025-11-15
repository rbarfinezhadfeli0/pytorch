# Documentation: `c10/util/overflows.h`

## File Metadata

- **Path**: `c10/util/overflows.h`
- **Size**: 3,467 bytes (3.39 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/complex.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace c10 {
// In some versions of MSVC, there will be a compiler error when building.
// C4146: unary minus operator applied to unsigned type, result still unsigned
// C4804: unsafe use of type 'bool' in operation
// It can be addressed by disabling the following warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)
#pragma warning(disable : 4804)
#pragma warning(disable : 4018)
#endif

// The overflow checks may involve float to int conversion which may
// trigger precision loss warning. Re-enable the warning once the code
// is fixed. See T58053069.
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

// bool can be converted to any type.
// Without specializing on bool, in pytorch_linux_trusty_py2_7_9_build:
// `error: comparison of constant '255' with boolean expression is always false`
// for `f > limit::max()` below
template <typename To, typename From>
std::enable_if_t<std::is_same_v<From, bool>, bool> overflows(
    From /*f*/,
    bool strict_unsigned [[maybe_unused]] = false) {
  return false;
}

// skip isnan and isinf check for integral types
template <typename To, typename From>
std::enable_if_t<std::is_integral_v<From> && !std::is_same_v<From, bool>, bool>
overflows(From f, bool strict_unsigned = false) {
  using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
  if constexpr (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    // allow for negative numbers to wrap using two's complement arithmetic.
    // For example, with uint8, this allows for `a - b` to be treated as
    // `a + 255 * b`.
    if (!strict_unsigned) {
      return greater_than_max<To>(f) ||
          (c10::is_negative(f) &&
           -static_cast<uint64_t>(f) > static_cast<uint64_t>(limit::max()));
    }
  }
  return c10::less_than_lowest<To>(f) || greater_than_max<To>(f);
}

template <typename To, typename From>
std::enable_if_t<std::is_floating_point_v<From>, bool> overflows(
    From f,
    bool strict_unsigned [[maybe_unused]] = false) {
  using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
  if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
    return false;
  }
  if (!limit::has_quiet_NaN && (f != f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

C10_CLANG_DIAGNOSTIC_POP()

#ifdef _MSC_VER
#pragma warning(pop)
#endif

template <typename To, typename From>
std::enable_if_t<is_complex<From>::value, bool> overflows(
    From f,
    bool strict_unsigned = false) {
  // casts from complex to real are considered to overflow if the
  // imaginary component is non-zero
  if (!is_complex<To>::value && f.imag() != 0) {
    return true;
  }
  // Check for overflow componentwise
  // (Technically, the imag overflow check is guaranteed to be false
  // when !is_complex<To>, but any optimizer worth its salt will be
  // able to figure it out.)
  return overflows<
             typename scalar_value_type<To>::type,
             typename From::value_type>(f.real(), strict_unsigned) ||
      overflows<
             typename scalar_value_type<To>::type,
             typename From::value_type>(f.imag(), strict_unsigned);
}
} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

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

- `c10/macros/Macros.h`
- `c10/util/TypeSafeSignMath.h`
- `c10/util/complex.h`
- `cmath`
- `limits`
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

- **File Documentation**: `overflows.h_docs.md`
- **Keyword Index**: `overflows.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
