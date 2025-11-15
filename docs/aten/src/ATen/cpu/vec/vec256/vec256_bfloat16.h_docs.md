# Documentation: `aten/src/ATen/cpu/vec/vec256/vec256_bfloat16.h`

## File Metadata

- **Path**: `aten/src/ATen/cpu/vec/vec256/vec256_bfloat16.h`
- **Size**: 8,807 bytes (8.60 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec256/vec256_16bit_float.h>
#include <c10/util/irange.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

template <>
struct is_vec_specialized_for<BFloat16> : std::bool_constant<true> {};

template <>
class Vectorized<BFloat16> : public Vectorized16<BFloat16> {
 public:
  using Vectorized16::Vectorized16;

  using value_type = BFloat16;

  Vectorized<BFloat16> frac() const;

  Vectorized<BFloat16> eq(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ne(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> gt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ge(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> lt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> le(const Vectorized<BFloat16>& other) const;
};

Vectorized<BFloat16> inline operator+(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_add_ps(x, y);
  });
}
Vectorized<BFloat16> inline operator-(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_sub_ps(x, y);
  });
}
Vectorized<BFloat16> inline operator*(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_mul_ps(x, y);
  });
}
Vectorized<BFloat16> inline operator/(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_div_ps(x, y);
  });
}
Vectorized<BFloat16> inline operator&(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return _mm256_and_si256(a, b);
}
Vectorized<BFloat16> inline operator|(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return _mm256_or_si256(a, b);
}
Vectorized<BFloat16> inline operator^(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return _mm256_xor_si256(a, b);
}

inline Vectorized<BFloat16> Vectorized<BFloat16>::eq(
    const Vectorized<BFloat16>& other) const {
  return (*this == other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::ne(
    const Vectorized<BFloat16>& other) const {
  return (*this != other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::gt(
    const Vectorized<BFloat16>& other) const {
  return (*this > other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::ge(
    const Vectorized<BFloat16>& other) const {
  return (*this >= other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::lt(
    const Vectorized<BFloat16>& other) const {
  return (*this < other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::le(
    const Vectorized<BFloat16>& other) const {
  return (*this <= other) & Vectorized<BFloat16>(1.0f);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<BFloat16> Vectorized<BFloat16>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<BFloat16> inline maximum(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  auto max_lo = _mm256_max_ps(a_lo, b_lo);
  auto max_hi = _mm256_max_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm256_or_ps(max_lo, nan_lo);
  auto o2 = _mm256_or_ps(max_hi, nan_hi);
  return cvtfp32_bf16(o1, o2);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<BFloat16> inline minimum(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  auto min_lo = _mm256_min_ps(a_lo, b_lo);
  auto min_hi = _mm256_min_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm256_or_ps(min_lo, nan_lo);
  auto o2 = _mm256_or_ps(min_hi, nan_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min,
    const Vectorized<BFloat16>& max) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  __m256 max_lo, max_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(min), min_lo, min_hi);
  cvtbf16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, _mm256_max_ps(min_lo, a_lo));
  auto o2 = _mm256_min_ps(max_hi, _mm256_max_ps(min_hi, a_hi));
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_max(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& max) {
  __m256 a_lo, a_hi;
  __m256 max_lo, max_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, a_lo);
  auto o2 = _mm256_min_ps(max_hi, a_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_min(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(min), min_lo, min_hi);
  auto o1 = _mm256_max_ps(min_lo, a_lo);
  auto o2 = _mm256_max_ps(min_hi, a_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
inline void convert(const BFloat16* src, BFloat16* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<BFloat16>::size());
       i += Vectorized<BFloat16>::size()) {
    auto vsrc =
        _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>((void*)(dst + i)), vsrc);
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <>
inline void convert(const float* src, BFloat16* dst, int64_t n) {
  int64_t i;
  for (i = 0; i + Vectorized<BFloat16>::size() <= n;
       i += Vectorized<BFloat16>::size()) {
    __m256 a = _mm256_loadu_ps(&src[i]);
    __m256 b = _mm256_loadu_ps(&src[i + 8]);

    __m256i bf = cvtfp32_bf16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
inline void convert(const double* src, BFloat16* dst, int64_t n) {
  auto load_float = [](const double* src) -> __m256 {
    // Load one float vector from an array of doubles
    __m128 a = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
    __m128 b = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  };

  int64_t i;
  for (i = 0; i + Vectorized<BFloat16>::size() <= n;
       i += Vectorized<BFloat16>::size()) {
    __m256 a = load_float(&src[i]);
    __m256 b = load_float(&src[i + 8]);

    __m256i bf = cvtfp32_bf16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
Vectorized<BFloat16> inline fmadd(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b,
    const Vectorized<BFloat16>& c) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  __m256 c_lo, c_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  cvtbf16_fp32(__m256i(c), c_lo, c_hi);
  auto o1 = _mm256_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm256_fmadd_ps(a_hi, b_hi, c_hi);
  return cvtfp32_bf16(o1, o2);
}

CONVERT_VECTORIZED_INIT(BFloat16, bfloat16)
LOAD_FP32_VECTORIZED_INIT(BFloat16, bf16)

#else // defined(CPU_CAPABILITY_AVX2)

#if !(                                                                      \
    defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__) && \
    !defined(CPU_CAPABILITY_SVE256))
CONVERT_NON_VECTORIZED_INIT(BFloat16, bfloat16)
#endif

LOAD_FP32_NON_VECTORIZED_INIT(BFloat16, bf16)
#endif // defined(CPU_CAPABILITY_AVX2)
} // namespace CPU_CAPABILITY
} // namespace at::vec

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 31 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `CPU_CAPABILITY`, `at`

**Classes/Structs**: `is_vec_specialized_for`, `Vectorized`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cpu/vec/vec256`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cpu/vec/vec256/vec256_16bit_float.h`
- `c10/util/irange.h`


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

Files in the same folder (`aten/src/ATen/cpu/vec/vec256`):

- [`vec256_complex_double.h_docs.md`](./vec256_complex_double.h_docs.md)
- [`missing_vst1_neon.h_docs.md`](./missing_vst1_neon.h_docs.md)
- [`vec256.h_docs.md`](./vec256.h_docs.md)
- [`vec256_16bit_float.h_docs.md`](./vec256_16bit_float.h_docs.md)
- [`vec256_float.h_docs.md`](./vec256_float.h_docs.md)
- [`missing_vld1_neon.h_docs.md`](./missing_vld1_neon.h_docs.md)
- [`vec256_convert.h_docs.md`](./vec256_convert.h_docs.md)
- [`vec256_mask.h_docs.md`](./vec256_mask.h_docs.md)
- [`vec256_half.h_docs.md`](./vec256_half.h_docs.md)


## Cross-References

- **File Documentation**: `vec256_bfloat16.h_docs.md`
- **Keyword Index**: `vec256_bfloat16.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
