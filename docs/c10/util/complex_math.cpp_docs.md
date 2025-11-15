# Documentation: `c10/util/complex_math.cpp`

## File Metadata

- **Path**: `c10/util/complex_math.cpp`
- **Size**: 3,157 bytes (3.08 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/complex.h>

// Note [ Complex Square root in libc++]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In libc++ complex square root is computed using polar form
// This is a reasonably fast algorithm, but can result in significant
// numerical errors when arg is close to 0, pi/2, pi, or 3pi/4
// In that case provide a more conservative implementation which is
// slower but less prone to those kinds of errors
// In libstdc++ complex square root yield invalid results
// for -x-0.0j unless C99 csqrt/csqrtf fallbacks are used

#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))

namespace {
template <typename T>
c10::complex<T> compute_csqrt(const c10::complex<T>& z) {
  constexpr auto half = T(.5);

  // Trust standard library to correctly handle infs and NaNs
  if (std::isinf(z.real()) || std::isinf(z.imag()) || std::isnan(z.real()) ||
      std::isnan(z.imag())) {
    return static_cast<c10::complex<T>>(
        std::sqrt(static_cast<std::complex<T>>(z)));
  }

  // Special case for square root of pure imaginary values
  if (z.real() == T(0)) {
    if (z.imag() == T(0)) {
      return c10::complex<T>(T(0), z.imag());
    }
    auto v = std::sqrt(half * std::abs(z.imag()));
    return c10::complex<T>(v, std::copysign(v, z.imag()));
  }

  // At this point, z is non-zero and finite
  if (z.real() >= 0.0) {
    auto t = std::sqrt((z.real() + std::abs(z)) * half);
    return c10::complex<T>(t, half * (z.imag() / t));
  }

  auto t = std::sqrt((-z.real() + std::abs(z)) * half);
  return c10::complex<T>(
      half * std::abs(z.imag() / t), std::copysign(t, z.imag()));
}

// Compute complex arccosine using formula from W. Kahan
// "Branch Cuts for Complex Elementary Functions" 1986 paper:
// cacos(z).re = 2*atan2(sqrt(1-z).re(), sqrt(1+z).re())
// cacos(z).im = asinh((sqrt(conj(1+z))*sqrt(1-z)).im())
template <typename T>
c10::complex<T> compute_cacos(const c10::complex<T>& z) {
  auto constexpr one = T(1);
  // Trust standard library to correctly handle infs and NaNs
  if (std::isinf(z.real()) || std::isinf(z.imag()) || std::isnan(z.real()) ||
      std::isnan(z.imag())) {
    return static_cast<c10::complex<T>>(
        std::acos(static_cast<std::complex<T>>(z)));
  }
  auto a = compute_csqrt(c10::complex<T>(one - z.real(), -z.imag()));
  auto b = compute_csqrt(c10::complex<T>(one + z.real(), z.imag()));
  auto c = compute_csqrt(c10::complex<T>(one + z.real(), -z.imag()));
  auto r = T(2) * std::atan2(a.real(), b.real());
  // Explicitly unroll (a*c).imag()
  auto i = std::asinh(a.real() * c.imag() + a.imag() * c.real());
  return c10::complex<T>(r, i);
}
} // anonymous namespace

namespace c10_complex_math {
namespace _detail {
c10::complex<float> sqrt(const c10::complex<float>& in) {
  return compute_csqrt(in);
}

c10::complex<double> sqrt(const c10::complex<double>& in) {
  return compute_csqrt(in);
}

c10::complex<float> acos(const c10::complex<float>& in) {
  return compute_cacos(in);
}

c10::complex<double> acos(const c10::complex<double>& in) {
  return compute_cacos(in);
}

} // namespace _detail
} // namespace c10_complex_math
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `_detail`, `c10_complex_math`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/complex.h`


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

- **File Documentation**: `complex_math.cpp_docs.md`
- **Keyword Index**: `complex_math.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
