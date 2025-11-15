# Documentation: `c10/util/MathConstants.h`

## File Metadata

- **Path**: `c10/util/MathConstants.h`
- **Size**: 3,690 bytes (3.60 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

namespace c10 {
// TODO: Replace me with inline constexpr variable when C++17 becomes available
namespace detail {
template <typename T>
C10_HOST_DEVICE inline constexpr T e() {
  return static_cast<T>(2.718281828459045235360287471352662);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T euler() {
  return static_cast<T>(0.577215664901532860606512090082402);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_1_pi() {
  return static_cast<T>(0.318309886183790671537767526745028);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_1_sqrt_pi() {
  return static_cast<T>(0.564189583547756286948079451560772);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_sqrt_2() {
  return static_cast<T>(0.707106781186547524400844362104849);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_sqrt_3() {
  return static_cast<T>(0.577350269189625764509148780501957);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T golden_ratio() {
  return static_cast<T>(1.618033988749894848204586834365638);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T ln_10() {
  return static_cast<T>(2.302585092994045684017991454684364);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T ln_2() {
  return static_cast<T>(0.693147180559945309417232121458176);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T log_10_e() {
  return static_cast<T>(0.434294481903251827651128918916605);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T log_2_e() {
  return static_cast<T>(1.442695040888963407359924681001892);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T pi() {
  return static_cast<T>(3.141592653589793238462643383279502);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T sqrt_2() {
  return static_cast<T>(1.414213562373095048801688724209698);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T sqrt_3() {
  return static_cast<T>(1.732050807568877293527446341505872);
}

template <>
C10_HOST_DEVICE inline constexpr BFloat16 pi<BFloat16>() {
  // According to
  // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Special_values
  // pi is encoded as 4049
  return BFloat16(0x4049, BFloat16::from_bits());
}

template <>
C10_HOST_DEVICE inline constexpr Half pi<Half>() {
  return Half(0x4248, Half::from_bits());
}
} // namespace detail

template <typename T>
constexpr T e = c10::detail::e<T>();

template <typename T>
constexpr T euler = c10::detail::euler<T>();

template <typename T>
constexpr T frac_1_pi = c10::detail::frac_1_pi<T>();

template <typename T>
constexpr T frac_1_sqrt_pi = c10::detail::frac_1_sqrt_pi<T>();

template <typename T>
constexpr T frac_sqrt_2 = c10::detail::frac_sqrt_2<T>();

template <typename T>
constexpr T frac_sqrt_3 = c10::detail::frac_sqrt_3<T>();

template <typename T>
constexpr T golden_ratio = c10::detail::golden_ratio<T>();

template <typename T>
constexpr T ln_10 = c10::detail::ln_10<T>();

template <typename T>
constexpr T ln_2 = c10::detail::ln_2<T>();

template <typename T>
constexpr T log_10_e = c10::detail::log_10_e<T>();

template <typename T>
constexpr T log_2_e = c10::detail::log_2_e<T>();

template <typename T>
constexpr T pi = c10::detail::pi<T>();

template <typename T>
constexpr T sqrt_2 = c10::detail::sqrt_2<T>();

template <typename T>
constexpr T sqrt_3 = c10::detail::sqrt_3<T>();
} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Macros.h`
- `c10/util/BFloat16.h`
- `c10/util/Half.h`


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

- **File Documentation**: `MathConstants.h_docs.md`
- **Keyword Index**: `MathConstants.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
