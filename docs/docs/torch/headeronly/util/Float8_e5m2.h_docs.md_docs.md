# Documentation: `docs/torch/headeronly/util/Float8_e5m2.h_docs.md`

## File Metadata

- **Path**: `docs/torch/headeronly/util/Float8_e5m2.h_docs.md`
- **Size**: 17,261 bytes (16.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/headeronly/util/Float8_e5m2.h`

## File Metadata

- **Path**: `torch/headeronly/util/Float8_e5m2.h`
- **Size**: 14,852 bytes (14.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

/// Defines the Float8_e5m2 type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration:
/// s eeeee mm
/// 1 sign bit
/// 5 exponent bits
/// 2 mantissa bits
/// bias = 15
///
/// Implementation based on the paper https://arxiv.org/pdf/2209.05433.pdf
/// and inspired by Half implementation from pytorch/c10/util/Half.h

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Half.h>

#include <limits>

namespace c10 {

struct alignas(1) Float8_e5m2 {
  uint8_t x;

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  Float8_e5m2() = default;

  constexpr C10_HOST_DEVICE Float8_e5m2(uint8_t bits, from_bits_t /*unused*/)
      : x(bits) {}
  inline C10_HOST_DEVICE Float8_e5m2(float value);
  inline C10_HOST_DEVICE operator float() const;
  inline C10_HOST_DEVICE bool isnan() const;
  inline C10_HOST_DEVICE bool isinf() const;
};

inline std::ostream& operator<<(std::ostream& out, const Float8_e5m2& value) {
  out << (float)value;
  return out;
}

namespace detail {

/*
 * Convert a 8-bit floating-point number in fp8 E5M2 format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline C10_HOST_DEVICE float fp8e5m2_to_fp32_value(uint8_t input) {
  /*
   * Extend the fp8 E5M2 number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+----+---+-----------------------------+
   *      | S |EEEEE|MM|0000 0000 0000 0000 0000 0000|
   *      +---+----+---+-----------------------------+
   * Bits  31 26-30 24-25          0-23
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  uint16_t half_representation = input;
  half_representation <<= 8;
  return fp16_ieee_to_fp32_value(half_representation);
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
inline C10_HOST_DEVICE uint8_t fp8e5m2_from_fp32_value(float f) {
  /*
   * Binary representation of fp32 infinity
   * 0 11111111 00000000000000000000000
   */
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;

  /*
   * Binary representation of 65536.0f, which is the first value
   * not representable in fp8e5m2 range:
   * 0 11111 00 - fp8e5m2
   * 0 10001111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e5m2 normal range
   * into denorm representation
   * magic number: ((127 - 15) + (23 - 2) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint8_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    // NaN - all exponent and mantissa bits set to 1
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (f_bits < (UINT32_C(113) << 23)) {
      // Input number is smaller than 2^(-14), which is the smallest
      // fp8e5m2 normal number
      f_bits =
          fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      // resulting mantissa is odd
      uint32_t mant_odd = (f_bits >> 21) & 1;

      // update exponent, rounding bias part 1
      f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;

      // rounding bias part 2
      f_bits += mant_odd;

      // take the bits!
      result = static_cast<uint8_t>(f_bits >> 21);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

} // namespace detail

// -------- below is copied from c10/util/Float8_e5m2-inl.h --------//
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

#define EXP_WIDTH_FP8 5
#define MAN_WIDTH_FP8 2
#define EXP_BIAS_FP8 15

/// Constructors

inline C10_HOST_DEVICE Float8_e5m2::Float8_e5m2(float value)
    : x(detail::fp8e5m2_from_fp32_value(value)) {}

/// Implicit conversions

inline C10_HOST_DEVICE Float8_e5m2::operator float() const {
  return detail::fp8e5m2_to_fp32_value(x);
}

/// Special values helpers

inline C10_HOST_DEVICE bool Float8_e5m2::isnan() const {
  return (x & 0b01111111) > 0b01111100;
}

inline C10_HOST_DEVICE bool Float8_e5m2::isinf() const {
  return (x & 0b01111111) == 0b01111100;
}

/// Arithmetic

inline C10_HOST_DEVICE Float8_e5m2
operator+(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e5m2
operator-(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e5m2
operator*(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e5m2 operator/(
    const Float8_e5m2& a,
    const Float8_e5m2& b) __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e5m2 operator-(const Float8_e5m2& a) {
  return -static_cast<float>(a);
}

inline C10_HOST_DEVICE Float8_e5m2& operator+=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2& operator-=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2& operator*=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2& operator/=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(Float8_e5m2 a, float b) {
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(Float8_e5m2 a, float b) {
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(Float8_e5m2 a, float b) {
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(Float8_e5m2 a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, Float8_e5m2 b) {
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, Float8_e5m2 b) {
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, Float8_e5m2 b) {
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, Float8_e5m2 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const Float8_e5m2& b) {
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const Float8_e5m2& b) {
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const Float8_e5m2& b) {
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const Float8_e5m2& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(Float8_e5m2 a, double b) {
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(Float8_e5m2 a, double b) {
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(Float8_e5m2 a, double b) {
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(Float8_e5m2 a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline C10_HOST_DEVICE double operator+(double a, Float8_e5m2 b) {
  return a + static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator-(double a, Float8_e5m2 b) {
  return a - static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator*(double a, Float8_e5m2 b) {
  return a * static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator/(double a, Float8_e5m2 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline C10_HOST_DEVICE Float8_e5m2 operator+(Float8_e5m2 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a + static_cast<Float8_e5m2>(b);
}
inline C10_HOST_DEVICE Float8_e5m2 operator-(Float8_e5m2 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a - static_cast<Float8_e5m2>(b);
}
inline C10_HOST_DEVICE Float8_e5m2 operator*(Float8_e5m2 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a * static_cast<Float8_e5m2>(b);
}
inline C10_HOST_DEVICE Float8_e5m2 operator/(Float8_e5m2 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a / static_cast<Float8_e5m2>(b);
}

inline C10_HOST_DEVICE Float8_e5m2 operator+(int a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) + b;
}
inline C10_HOST_DEVICE Float8_e5m2 operator-(int a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) - b;
}
inline C10_HOST_DEVICE Float8_e5m2 operator*(int a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) * b;
}
inline C10_HOST_DEVICE Float8_e5m2 operator/(int a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) / b;
}

//// Arithmetic with int64_t

inline C10_HOST_DEVICE Float8_e5m2 operator+(Float8_e5m2 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a + static_cast<Float8_e5m2>(b);
}
inline C10_HOST_DEVICE Float8_e5m2 operator-(Float8_e5m2 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a - static_cast<Float8_e5m2>(b);
}
inline C10_HOST_DEVICE Float8_e5m2 operator*(Float8_e5m2 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a * static_cast<Float8_e5m2>(b);
}
inline C10_HOST_DEVICE Float8_e5m2 operator/(Float8_e5m2 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a / static_cast<Float8_e5m2>(b);
}

inline C10_HOST_DEVICE Float8_e5m2 operator+(int64_t a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) + b;
}
inline C10_HOST_DEVICE Float8_e5m2 operator-(int64_t a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) - b;
}
inline C10_HOST_DEVICE Float8_e5m2 operator*(int64_t a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) * b;
}
inline C10_HOST_DEVICE Float8_e5m2 operator/(int64_t a, Float8_e5m2 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<Float8_e5m2>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Float8_e5m2 to float.
C10_CLANG_DIAGNOSTIC_POP()
} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::Float8_e5m2;
using c10::operator<<;
using c10::operator+;
using c10::operator-;
using c10::operator*;
using c10::operator/;
using c10::operator+=;
using c10::operator-=;
using c10::operator*=;
using c10::operator/=;

namespace detail {
using c10::detail::fp8e5m2_from_fp32_value;
using c10::detail::fp8e5m2_to_fp32_value;
} // namespace detail
HIDDEN_NAMESPACE_END(torch, headeronly)

namespace std {

template <>
class numeric_limits<c10::Float8_e5m2> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = true;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 3;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 2;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr c10::Float8_e5m2 min() {
    return c10::Float8_e5m2(0x4, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 max() {
    return c10::Float8_e5m2(0x7B, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 lowest() {
    return c10::Float8_e5m2(0xFB, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 epsilon() {
    return c10::Float8_e5m2(0x34, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 round_error() {
    return c10::Float8_e5m2(0x38, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 infinity() {
    return c10::Float8_e5m2(0x7C, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 quiet_NaN() {
    return c10::Float8_e5m2(0x7F, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 denorm_min() {
    return c10::Float8_e5m2(0x01, c10::Float8_e5m2::from_bits());
  }
};

} // namespace std

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 24 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `std`, `c10`

**Classes/Structs**: `alignas`, `from_bits_t`, `numeric_limits`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/headeronly/util`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/headeronly/macros/Macros.h`
- `torch/headeronly/util/Half.h`
- `limits`


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
- [`floating_point_utils.h_docs.md`](./floating_point_utils.h_docs.md)
- [`shim_utils.h_docs.md`](./shim_utils.h_docs.md)
- [`TypeList.h_docs.md`](./TypeList.h_docs.md)
- [`Float4_e2m1fn_x2.h_docs.md`](./Float4_e2m1fn_x2.h_docs.md)
- [`Float8_e8m0fnu.h_docs.md`](./Float8_e8m0fnu.h_docs.md)
- [`BFloat16.h_docs.md`](./BFloat16.h_docs.md)
- [`Float8_fnuz_cvt.h_docs.md`](./Float8_fnuz_cvt.h_docs.md)


## Cross-References

- **File Documentation**: `Float8_e5m2.h_docs.md`
- **Keyword Index**: `Float8_e5m2.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/headeronly/util`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/headeronly/util`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/headeronly/util`):

- [`quint8.h_docs.md_docs.md`](./quint8.h_docs.md_docs.md)
- [`TypeTraits.h_kw.md_docs.md`](./TypeTraits.h_kw.md_docs.md)
- [`Half.h_kw.md_docs.md`](./Half.h_kw.md_docs.md)
- [`TypeSafeSignMath.h_kw.md_docs.md`](./TypeSafeSignMath.h_kw.md_docs.md)
- [`qint32.h_docs.md_docs.md`](./qint32.h_docs.md_docs.md)
- [`Float8_e4m3fnuz.h_kw.md_docs.md`](./Float8_e4m3fnuz.h_kw.md_docs.md)
- [`Exception.h_docs.md_docs.md`](./Exception.h_docs.md_docs.md)
- [`quint8.h_kw.md_docs.md`](./quint8.h_kw.md_docs.md)
- [`quint2x4.h_docs.md_docs.md`](./quint2x4.h_docs.md_docs.md)
- [`quint4x2.h_docs.md_docs.md`](./quint4x2.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Float8_e5m2.h_docs.md_docs.md`
- **Keyword Index**: `Float8_e5m2.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
