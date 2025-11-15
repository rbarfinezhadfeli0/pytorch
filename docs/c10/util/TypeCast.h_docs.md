# Documentation: `c10/util/TypeCast.h`

## File Metadata

- **Path**: `c10/util/TypeCast.h`
- **Size**: 6,662 bytes (6.51 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Float8_e8m0fnu.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <c10/util/overflows.h>

#include <type_traits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

template <typename dest_t, typename src_t>
struct needs_real {
  constexpr static bool value =
      (is_complex<src_t>::value && !is_complex<dest_t>::value);
};

template <bool, typename src_t>
struct maybe_real {
  C10_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

template <typename src_t>
struct maybe_real<true, src_t> {
  C10_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    return src.real();
  }
};

template <bool, typename src_t>
struct maybe_bool {
  C10_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

template <typename src_t>
struct maybe_bool<true, src_t> {
  C10_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    // Don't use bool operator so as to also compile for ComplexHalf.
    return src.real() || src.imag();
  }
};

// Note: deliberately ignores undefined behavior, consistent with NumPy.
// PyTorch's type conversions can cause a variety of undefined behavior,
// including float to integral overflow and signed to unsigned integer overflow.
// Some of this undefined behavior is addressed below.
template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline dest_t apply(
      src_t src) {
    constexpr bool real = needs_real<dest_t, src_t>::value;
    auto r = maybe_real<real, src_t>::apply(src);
    return static_cast<dest_t>(r);
  }
};

// Partial template specialization for casting to bool.
// Need to handle complex types separately, as we don't
// simply want to cast the real part to bool.
template <typename src_t>
struct static_cast_with_inter_type<bool, src_t> {
  C10_HOST_DEVICE static inline bool apply(src_t src) {
    constexpr bool complex = needs_real<bool, src_t>::value;
    return static_cast<bool>(maybe_bool<complex, src_t>::apply(src));
  }
};

// Partial template instantiation for casting to uint8.
// Note: Converting from negative float values to unsigned integer types is
// undefined behavior in C++, and current CPU and GPU compilers exhibit
// divergent behavior. Casting from negative float values to signed
// integer types and then to unsigned integer types is not undefined,
// however, so this cast improves the consistency of type conversions
// to uint8 across compilers.
// Further note: Type conversions across compilers still have other undefined
// and divergent behavior.
template <typename src_t>
struct static_cast_with_inter_type<uint8_t, src_t> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline uint8_t apply(
      src_t src) {
    constexpr bool real = needs_real<uint8_t, src_t>::value;
    return static_cast<uint8_t>(
        static_cast<int64_t>(maybe_real<real, src_t>::apply(src)));
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::BFloat16> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::BFloat16 src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::Float8_e5m2> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::Float8_e5m2 src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e5m2fnuz> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::Float8_e5m2fnuz src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e4m3fn> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::Float8_e4m3fn src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e4m3fnuz> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::Float8_e4m3fnuz src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

// TODO(#146647): Can we make all these template specialization happen
// based off our apply macros?
template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e8m0fnu> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::Float8_e8m0fnu src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::Half> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::Half src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::complex<double>> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::complex<double> src) {
    return static_cast<c10::complex<c10::Half>>(
        static_cast<c10::complex<float>>(src));
  }
};

template <typename To, typename From>
C10_HOST_DEVICE To convert(From f) {
  return static_cast_with_inter_type<To, From>::apply(f);
}

// Define separately to avoid being inlined and prevent code-size bloat
[[noreturn]] C10_API void report_overflow(const char* name);

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  // Converting to bool can't overflow so we exclude this case from checking.
  if (!std::is_same_v<To, bool> && overflows<To, From>(f)) {
    report_overflow(name);
  }
  return convert<To, From>(f);
}

} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()

// Trigger tests for D25440771. TODO: Remove this line any time you want.

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `needs_real`, `maybe_real`, `maybe_real`, `maybe_bool`, `maybe_bool`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`, `static_cast_with_inter_type`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Macros.h`
- `c10/util/BFloat16.h`
- `c10/util/Float8_e4m3fn.h`
- `c10/util/Float8_e4m3fnuz.h`
- `c10/util/Float8_e5m2.h`
- `c10/util/Float8_e5m2fnuz.h`
- `c10/util/Float8_e8m0fnu.h`
- `c10/util/Half.h`
- `c10/util/complex.h`
- `c10/util/overflows.h`
- `type_traits`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `TypeCast.h_docs.md`
- **Keyword Index**: `TypeCast.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
