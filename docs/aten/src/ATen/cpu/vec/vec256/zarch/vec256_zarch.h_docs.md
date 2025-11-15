# Documentation: `aten/src/ATen/cpu/vec/vec256/zarch/vec256_zarch.h`

## File Metadata

- **Path**: `aten/src/ATen/cpu/vec/vec256/zarch/vec256_zarch.h`
- **Size**: 102,025 bytes (99.63 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#if defined(__clang__)
#include <sleef.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#include <sleef.h>
#include <vecintrin.h>
#endif
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/complex.h>

namespace at {
namespace vec {

// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

template <typename T>
constexpr bool is_zarch_implemented() {
  return (
      std::is_same_v<T, float> || std::is_same_v<T, double> ||
      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
      std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t> ||
      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
}

template <typename T>
constexpr bool is_zarch_implemented_quant() {
  return (
      std::is_same_v<T, c10::qint32> || std::is_same_v<T, c10::qint8> ||
      std::is_same_v<T, c10::quint8>);
}

template <typename T>
constexpr bool is_zarch_implemented_complex() {
  return std::is_same_v<T, c10::complex<float>> ||
      std::is_same_v<T, c10::complex<double>>;
}

constexpr int offset0 = 0;
constexpr int offset16 = 16;

template <int N>
struct VecBinaryType {
  using type __attribute__((vector_size(16))) = uintmax_t;
};

template <>
struct VecBinaryType<8> {
  using type = __attribute__((vector_size(16))) unsigned long long;
};

template <>
struct VecBinaryType<4> {
  using type = __attribute__((vector_size(16))) unsigned int;
};

template <>
struct VecBinaryType<2> {
  using type = __attribute__((vector_size(16))) unsigned short;
};

template <>
struct VecBinaryType<1> {
  using type = __attribute__((vector_size(16))) unsigned char;
};

template <typename T>
struct VecInnerType {
  using Type __attribute__((vector_size(16))) = T;
  using BinaryType = typename VecBinaryType<sizeof(T)>::type;
  using ElementType = T;
  static constexpr int size = 16 / sizeof(T);
};

// define for int64_t properly for load
template <>
struct VecInnerType<int64_t> {
  using Type = __attribute__((vector_size(16))) signed long long;
  using ElementType = signed long long;
  using BinaryType = typename VecBinaryType<sizeof(signed long long)>::type;
  static constexpr int size = 16 / sizeof(signed long long);
};

template <typename T>
using ZSimdVect = typename VecInnerType<T>::Type;
template <typename T>
using ZSimdVectBinary = typename VecInnerType<T>::BinaryType;
template <typename T>
using ZSimdVectElement = typename VecInnerType<T>::ElementType;

constexpr int blendChoiceInner(
    const uint64_t mask,
    const uint64_t half1 = 0xF,
    const uint64_t half2 = 0xF0) {
  uint64_t none = 0;
  uint64_t both = half1 | half2;
  // clamp it between 0 and both
  auto res_mask = mask & both;
  // return  (a._vec0, a._vec1)
  if (res_mask == none)
    return 0;
  // return (b._vec0,b._vec1)
  else if (res_mask == both)
    return 1;
  // return  (b._vec0, a._vec1)
  else if (res_mask == half1)
    return 2;
  // return  (a._vec0,b._vec1)
  else if (res_mask == half2)
    return 3;
  // return  (*_vec0,a._vec1)
  else if (res_mask > 0 && res_mask < half1)
    return 4;
  // return  (*_vec0,b._vec1)
  else if ((res_mask & half2) == half2)
    return 5;
  // return (a._vec0,*_vec1)
  else if ((res_mask & half1) == 0 && res_mask > half1)
    return 6;
  // return (b._vec0,*_vec1)
  else if ((res_mask & half1) == half1 && res_mask > half1)
    return 7;
  // return (*_vec0,*_vec1)
  return 8;
}

// it can be used to emulate blend faster
template <int Z>
constexpr int blendChoice(const uint64_t mask) {
  static_assert(Z < 1 || Z > 8, "not implemented");
  return blendChoiceInner(mask);
}

template <>
constexpr int blendChoice<1>(const uint64_t mask) {
  return blendChoiceInner(mask, 0x0000FFFF, 0xFFFF0000);
}

template <>
constexpr int blendChoice<2>(const uint64_t mask) {
  return blendChoiceInner(mask, 0x00FF, 0xFF00);
}

template <>
constexpr int blendChoice<4>(const uint64_t mask) {
  return blendChoiceInner(mask, 0xF, 0xF0);
}

template <>
constexpr int blendChoice<8>(const uint64_t mask) {
  // clamp it 0 and 0xF
  return blendChoiceInner(mask, 0x3, 0xC);
}

template <int N>
constexpr auto GetMask1(const uint64_t mask) {
  return typename VecBinaryType<N>::type{};
}

template <int N>
constexpr auto GetMask2(const uint64_t mask) {
  return typename VecBinaryType<N>::type{};
}

template <>
constexpr auto GetMask1<1>(const uint64_t mask) {
  constexpr uint8_t t = (int)0xFF;
  uint8_t g0 = (mask & 1) * t;
  uint8_t g1 = ((mask & 2) >> 1) * t;
  uint8_t g2 = ((mask & 4) >> 2) * t;
  uint8_t g3 = ((mask & 8) >> 3) * t;
  uint8_t g4 = ((mask & 16) >> 4) * t;
  uint8_t g5 = ((mask & 32) >> 5) * t;
  uint8_t g6 = ((mask & 64) >> 6) * t;
  uint8_t g7 = ((mask & 128) >> 7) * t;
  uint8_t g8 = ((mask & 256) >> 8) * t;
  uint8_t g9 = ((mask & 512) >> 9) * t;
  uint8_t g10 = ((mask & 1024) >> 10) * t;
  uint8_t g11 = ((mask & 2048) >> 11) * t;
  uint8_t g12 = ((mask & 4096) >> 12) * t;
  uint8_t g13 = ((mask & 8192) >> 13) * t;
  uint8_t g14 = ((mask & 16384) >> 14) * t;
  uint8_t g15 = ((mask & 32768) >> 15) * t;
  return (typename VecBinaryType<1>::type){
      g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15};
}

template <>
constexpr auto GetMask2<1>(const uint64_t mask) {
  uint64_t mask2 = (mask & 0xFFFFFFFF) >> 16;
  return GetMask1<1>(mask2);
}

template <>
constexpr auto GetMask1<2>(const uint64_t mask) {
  constexpr uint16_t t = (int)0xFFFF;
  uint16_t g0 = (mask & 1) * t;
  uint16_t g1 = ((mask & 2) >> 1) * t;
  uint16_t g2 = ((mask & 4) >> 2) * t;
  uint16_t g3 = ((mask & 8) >> 3) * t;
  uint16_t g4 = ((mask & 16) >> 4) * t;
  uint16_t g5 = ((mask & 32) >> 5) * t;
  uint16_t g6 = ((mask & 64) >> 6) * t;
  uint16_t g7 = ((mask & 128) >> 7) * t;
  return (typename VecBinaryType<2>::type){g0, g1, g2, g3, g4, g5, g6, g7};
}

template <>
constexpr auto GetMask2<2>(const uint64_t mask) {
  uint64_t mask2 = (mask & 0xFFFF) >> 8;
  return GetMask1<2>(mask2);
}

template <>
constexpr auto GetMask1<4>(const uint64_t mask) {
  uint32_t g0 = (mask & 1) * 0xffffffff;
  uint32_t g1 = ((mask & 2) >> 1) * 0xffffffff;
  uint32_t g2 = ((mask & 4) >> 2) * 0xffffffff;
  uint32_t g3 = ((mask & 8) >> 3) * 0xffffffff;
  return (typename VecBinaryType<4>::type){g0, g1, g2, g3};
}

template <>
constexpr auto GetMask2<4>(const uint64_t mask) {
  uint64_t mask2 = (mask & 0xFF) >> 4;
  return GetMask1<4>(mask2);
}

template <>
constexpr auto GetMask1<8>(const uint64_t mask) {
  uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
  uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
  return (typename VecBinaryType<8>::type){g0, g1};
}

template <>
constexpr auto GetMask2<8>(const uint64_t mask) {
  uint64_t mask2 = (mask & 0xF) >> 2;
  return GetMask1<8>(mask2);
}

template <int Z>
constexpr int maskForComplex(uint32_t mask) {
  return 0;
}

template <>
constexpr int maskForComplex<8>(uint32_t mask) {
  mask = mask & 0xF;
  int complex_mask = 0;
  if (mask & 1)
    complex_mask |= 3;
  if (mask & 2)
    complex_mask |= (3 << 2);
  if (mask & 4)
    complex_mask |= (3 << 4);
  if (mask & 8)
    complex_mask |= (3 << 6);
  return complex_mask;
}

template <>
constexpr int maskForComplex<16>(uint32_t mask) {
  mask = mask & 0x3;
  int complex_mask = 0;
  if (mask & 1)
    complex_mask |= 3;
  if (mask & 2)
    complex_mask |= (3 << 2);
  return complex_mask;
}

template <typename T = c10::complex<float>>
constexpr int blend_choice() {
  return 0xAA;
}

template <>
constexpr int blend_choice<c10::complex<double>>() {
  return 0x0A;
}

constexpr int64_t allbitset(int16_t x) {
  int64_t onex = 1;
  return (onex << x) - onex;
}

namespace { /* unnamed namespace */

ZSimdVect<float> vec_mergee(ZSimdVect<float> x, ZSimdVect<float> y) {
  constexpr ZSimdVectBinary<uint8_t> mergee_mask{
      0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27};
  return vec_perm(x, y, mergee_mask);
}

ZSimdVect<double> vec_mergee(ZSimdVect<double> x, ZSimdVect<double> y) {
  return vec_mergeh(x, y);
}

ZSimdVect<float> vec_mergeo(ZSimdVect<float> x, ZSimdVect<float> y) {
  constexpr ZSimdVectBinary<uint8_t> mergeo_mask{
      4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31};
  return vec_perm(x, y, mergeo_mask);
}

ZSimdVect<double> vec_mergeo(ZSimdVect<double> x, ZSimdVect<double> y) {
  return vec_mergel(x, y);
}

} /* unnamed namespace */

//
template <typename T>
constexpr auto GetBpermZeroMask() {
  return ZSimdVectBinary<uint8_t>{
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      96,
      64,
      32,
      0};
}

template <>
constexpr auto GetBpermZeroMask<double>() {
  return ZSimdVectBinary<uint8_t>{
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      64,
      0};
}

constexpr auto GetSwapMaskFloat() {
  return ZSimdVectBinary<uint8_t>{
      4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
}

template <typename T>
struct is_vec_specialized_for<T, std::enable_if_t<is_zarch_implemented<T>()>>
    : std::bool_constant<true> {};

template <typename T>
struct Vectorized<T, std::enable_if_t<is_zarch_implemented<T>()>> {
 public:
  using value_type = T;
  using vtype = ZSimdVect<T>;
  using vmaskType = ZSimdVectBinary<T>;
  using size_type = int;
  // because of gcc inconsistency for int64_t we are obliged to use this, not
  // value_type
  using ElementType = ZSimdVectElement<T>;
  using vinner_data = std::pair<vtype, vtype>;

 private:
  vtype _vec0;
  vtype _vec1;

 public:
  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(ElementType);
  }
  Vectorized() {}

  C10_ALWAYS_INLINE Vectorized(vtype v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(const vinner_data& v)
      : _vec0{v.first}, _vec1{v.second} {}
  C10_ALWAYS_INLINE Vectorized(vtype v1, vtype v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(T s)
      : _vec0{vec_splats((ElementType)s)}, _vec1{vec_splats((ElementType)s)} {}

  template <typename U, typename DUMMY = void>
  struct LoaduHelper {
    static Vectorized<T> C10_ALWAYS_INLINE
    loadu(const U* ptr, int count = size()) {
      __at_align__ ElementType tmp_values[size()] = {};
      std::memcpy(
          tmp_values, ptr, std::min(count, size()) * sizeof(ElementType));

      return {
          vec_xl(offset0, &(tmp_values[0])),
          vec_xl(offset16, &(tmp_values[0]))};
    }
  };

  template <typename DUMMY>
  struct LoaduHelper<ElementType, DUMMY> {
    static Vectorized<T> C10_ALWAYS_INLINE
    loadu(const ElementType* ptr, int count = size()) {
      if (count == size()) {
        return {vec_xl(offset0, ptr), vec_xl(offset16, ptr)};
      }

      __at_align__ ElementType tmp_values[size()] = {};
      std::memcpy(
          tmp_values, ptr, std::min(count, size()) * sizeof(ElementType));

      return {
          vec_xl(offset0, &(tmp_values[0])),
          vec_xl(offset16, &(tmp_values[0]))};
    }
  };

  template <typename U>
  static Vectorized<T> C10_ALWAYS_INLINE
  loadu(const U* ptr, int count = size()) {
    return LoaduHelper<U>::loadu(ptr, count);
  }

  template <typename U>
  static Vectorized<T> C10_ALWAYS_INLINE loadu_one_fourth(const U* ptr) {
    // load only first 8 bytes
    // only intended to be used with uint8_t
    return loadu(ptr, 8 / sizeof(ElementType));
  }

  template <typename U, typename DUMMY = void>
  struct StoreHelper {
    static void C10_ALWAYS_INLINE
    store(const Vectorized<T>& vec, U* ptr, int count = size()) {
      if (count > 0) {
        __at_align__ ElementType tmp_values[size()];
        vec_xst(vec._vec0, offset0, &(tmp_values[0]));
        vec_xst(vec._vec1, offset16, &(tmp_values[0]));
        std::memcpy(
            ptr, tmp_values, std::min(count, size()) * sizeof(ElementType));
      }
    }
  };

  template <typename DUMMY>
  struct StoreHelper<ElementType, DUMMY> {
    static void C10_ALWAYS_INLINE
    store(const Vectorized<T>& vec, ElementType* ptr, int count = size()) {
      if (count == size()) {
        vec_xst(vec._vec0, offset0, ptr);
        vec_xst(vec._vec1, offset16, ptr);
      } else if (count > 0) {
        __at_align__ ElementType tmp_values[size()];
        vec_xst(vec._vec0, offset0, &(tmp_values[0]));
        vec_xst(vec._vec1, offset16, &(tmp_values[0]));
        std::memcpy(
            ptr, tmp_values, std::min(count, size()) * sizeof(ElementType));
      }
    }
  };

  template <typename U>
  void C10_ALWAYS_INLINE store(U* ptr, int count = size()) const {
    return StoreHelper<U>::store(*this, ptr, count);
  }

  C10_ALWAYS_INLINE const vtype& vec0() const {
    return _vec0;
  }

  C10_ALWAYS_INLINE const vtype& vec1() const {
    return _vec1;
  }

  C10_ALWAYS_INLINE vinner_data data() const {
    return std::make_pair<>(_vec0, _vec1);
  }

  C10_ALWAYS_INLINE operator vinner_data() const {
    return data();
  }

  C10_ALWAYS_INLINE const vmaskType vecb0() const {
    return (vmaskType)_vec0;
  }
  C10_ALWAYS_INLINE const vmaskType vecb1() const {
    return (vmaskType)_vec1;
  }

  static Vectorized<T> C10_ALWAYS_INLINE blendv(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      const Vectorized<T>& mask) {
    return {
        vec_sel(a._vec0, b._vec0, mask.vecb0()),
        vec_sel(a._vec1, b._vec1, mask.vecb1())};
  }

  template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
  C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4)
      : _vec0{s1, s2}, _vec1{s3, s4} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 4), int> = 0>
  C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4, T s5, T s6, T s7, T s8)
      : _vec0{s1, s2, s3, s4}, _vec1{s5, s6, s7, s8} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 2), int> = 0>
  C10_ALWAYS_INLINE Vectorized(
      T s1,
      T s2,
      T s3,
      T s4,
      T s5,
      T s6,
      T s7,
      T s8,
      T s9,
      T s10,
      T s11,
      T s12,
      T s13,
      T s14,
      T s15,
      T s16)
      : _vec0{s1, s2, s3, s4, s5, s6, s7, s8},
        _vec1{s9, s10, s11, s12, s13, s14, s15, s16} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 1), int> = 0>
  C10_ALWAYS_INLINE Vectorized(
      T s1,
      T s2,
      T s3,
      T s4,
      T s5,
      T s6,
      T s7,
      T s8,
      T s9,
      T s10,
      T s11,
      T s12,
      T s13,
      T s14,
      T s15,
      T s16,
      T s17,
      T s18,
      T s19,
      T s20,
      T s21,
      T s22,
      T s23,
      T s24,
      T s25,
      T s26,
      T s27,
      T s28,
      T s29,
      T s30,
      T s31,
      T s32)
      : _vec0{s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16},
        _vec1{
            s17,
            s18,
            s19,
            s20,
            s21,
            s22,
            s23,
            s24,
            s25,
            s26,
            s27,
            s28,
            s29,
            s30,
            s31,
            s32} {}

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 8, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(base, base + step, base + 2 * step, base + 3 * step);
  }

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 4, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
  }

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 2, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
  }

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 1, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step,
        base + 16 * step,
        base + 17 * step,
        base + 18 * step,
        base + 19 * step,
        base + 20 * step,
        base + 21 * step,
        base + 22 * step,
        base + 23 * step,
        base + 24 * step,
        base + 25 * step,
        base + 26 * step,
        base + 27 * step,
        base + 28 * step,
        base + 29 * step,
        base + 30 * step,
        base + 31 * step);
  }

  // blend section
  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 0, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return a;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 1, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return b;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 2, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return {b._vec0, a._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 3, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return {a._vec0, b._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 4, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
    return {(vtype)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 5, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
    return {(vtype)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 6, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
    // generated masks
    return {a._vec0, (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 7, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
    // generated masks
    return {b._vec0, (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 8, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
    const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
    return {
        (vtype)vec_sel(a._vec0, b._vec0, mask_1st),
        (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int16_t Z, int16_t C>
  static inline std::enable_if_t<(Z >= C), Vectorized<T>> set_inner(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count) {
    return b;
  }

  template <int16_t Z, int16_t C>
  static inline std::enable_if_t<(Z < C), Vectorized<T>> set_inner(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count) {
    if (count == Z)
      return blend<allbitset(Z)>(a, b);
    else
      return set_inner<Z + 1, C>(a, b, count);
  }

  static Vectorized<T> set(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count = size()) {
    if (count == 0)
      return a;
    return set_inner<1, size()>(a, b, count);
  }

  const ElementType& operator[](int idx) const = delete;
  ElementType& operator[](int idx) = delete;

  Vectorized<T> _not() const {
    return {(vtype)vec_nor(vecb0(), vecb0()), (vtype)vec_nor(vecb1(), vecb1())};
  }

  Vectorized<T> C10_ALWAYS_INLINE eq(const Vectorized<T>& other) const {
    return (*this == other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE ne(const Vectorized<T>& other) const {
    return (*this != other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE gt(const Vectorized<T>& other) const {
    return (*this > other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE ge(const Vectorized<T>& other) const {
    return (*this >= other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE lt(const Vectorized<T>& other) const {
    return (*this < other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE le(const Vectorized<T>& other) const {
    return (*this <= other) & Vectorized<T>((T)1.0);
  }

  template <typename U = T, std::enable_if_t<!std::is_unsigned_v<U>, int> = 0>
  Vectorized<U> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  template <typename U = T, std::enable_if_t<std::is_unsigned_v<U>, int> = 0>
  Vectorized<U> C10_ALWAYS_INLINE abs() const {
    return {_vec0, _vec1};
  }

  Vectorized<T> C10_ALWAYS_INLINE neg() const {
    return {-_vec0, -_vec1};
  }

  Vectorized<T> isnan() const {
    auto x = *this;
    auto ret = (x == x);
    return ret._not();
  }

  bool has_inf_nan() const {
    for (const auto i : c10::irange(size() / 2)) {
      if (_isnan(_vec0[i]) || _isinf(_vec0[i])) {
        return true;
      }
    }
    for (const auto i : c10::irange(size() / 2)) {
      if (_isnan(_vec1[i]) || _isinf(_vec1[i])) {
        return true;
      }
    }
    return false;
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  Vectorized<U> angle() const {
    auto tmp = blendv(
        Vectorized<U>(0), Vectorized<U>(c10::pi<U>), *this < Vectorized<U>(0));
    return blendv(tmp, *this, isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  Vectorized<U> angle() const {
    return blendv(
        Vectorized<U>(0), Vectorized<U>(c10::pi<U>), *this < Vectorized<U>(0));
  }

  Vectorized<T> real() const {
    return *this;
  }
  Vectorized<T> imag() const {
    return Vectorized<T>{0};
  }
  Vectorized<T> conj() const {
    return *this;
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  int zero_mask() const {
    auto cmp = (*this == Vectorized<U>(0));
    constexpr auto mask_zero_bits = GetBpermZeroMask<U>();
    ZSimdVectBinary<uint64_t> result0 =
        vec_bperm_u128((ZSimdVectBinary<uint8_t>)cmp.vecb0(), mask_zero_bits);
    ZSimdVectBinary<uint64_t> result1 =
        vec_bperm_u128((ZSimdVectBinary<uint8_t>)cmp.vecb1(), mask_zero_bits);
    return (result0[0] | (result1[0] << (size() / 2)));
  }

  Vectorized<T> C10_ALWAYS_INLINE floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE round() const {
    return {vec_round(_vec0), vec_round(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE rint() const {
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE frac() const {
    return *this - trunc();
  }

  Vectorized<T> C10_ALWAYS_INLINE sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  Vectorized<T> C10_ALWAYS_INLINE reciprocal() const {
    return Vectorized<T>((T)1) / (*this);
  }
  Vectorized<T> C10_ALWAYS_INLINE rsqrt() const {
    return sqrt().reciprocal();
  }

  template <typename U = T, std::enable_if_t<std::is_same_v<U, float>, int> = 0>
  inline Vectorized<T> mapOrdinary(float (*const f)(float)) const {
    float a00 = f(_vec0[0]);
    float a01 = f(_vec0[1]);
    float a02 = f(_vec0[2]);
    float a03 = f(_vec0[3]);
    float a10 = f(_vec1[0]);
    float a11 = f(_vec1[1]);
    float a12 = f(_vec1[2]);
    float a13 = f(_vec1[3]);
    return Vectorized<T>{a00, a01, a02, a03, a10, a11, a12, a13};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same_v<U, double>, int> = 0>
  inline Vectorized<T> mapOrdinary(double (*const f)(double)) const {
    return Vectorized<T>(f(_vec0[0]), f(_vec0[1]), f(_vec1[0]), f(_vec1[1]));
  }

  template <typename U = T, std::enable_if_t<std::is_same_v<U, float>, int> = 0>
  inline Vectorized<T> mapOrdinary(
      float (*const f)(float, float),
      const Vectorized<T>& b) const {
    float a00 = f(_vec0[0], b._vec0[0]);
    float a01 = f(_vec0[1], b._vec0[1]);
    float a02 = f(_vec0[2], b._vec0[2]);
    float a03 = f(_vec0[3], b._vec0[3]);
    float a10 = f(_vec1[0], b._vec1[0]);
    float a11 = f(_vec1[1], b._vec1[1]);
    float a12 = f(_vec1[2], b._vec1[2]);
    float a13 = f(_vec1[3], b._vec1[3]);
    return Vectorized<T>{a00, a01, a02, a03, a10, a11, a12, a13};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same_v<U, double>, int> = 0>
  inline Vectorized<T> mapOrdinary(
      double (*const f)(double, double),
      const Vectorized<T>& b) const {
    return Vectorized<T>(
        f(_vec0[0], b._vec0[0]),
        f(_vec0[1], b._vec0[1]),
        f(_vec1[0], b._vec1[0]),
        f(_vec1[1], b._vec1[1]));
  }

  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same_v<U, float>, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d) const {
    vtype a0 = f(_vec0);
    vtype a1 = f(_vec1);
    return Vectorized<T>{a0, a1};
  }

  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same_v<U, double>, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d) const {
    return Vectorized<T>(d(_vec0), d(_vec1));
  }

  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same_v<U, float>, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d, const Vectorized<T>& b)
      const {
    vtype a0 = f(_vec0, b._vec0);
    vtype a1 = f(_vec1, b._vec1);
    return Vectorized<T>{a0, a1};
  }

  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same_v<U, double>, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d, const Vectorized<T>& b)
      const {
    return Vectorized<T>(d(_vec0, b._vec0), d(_vec1, b._vec1));
  }

  Vectorized<T> acos() const {
    return mapSleef(Sleef_acosf4_u10, Sleef_acosd2_u10);
  }
  Vectorized<T> asin() const {
    return mapSleef(Sleef_asinf4_u10, Sleef_asind2_u10);
  }
  Vectorized<T> atan() const {
    return mapSleef(Sleef_atanf4_u10, Sleef_atand2_u10);
  }
  Vectorized<T> atanh() const {
    return mapSleef(Sleef_atanhf4_u10, Sleef_atanhd2_u10);
  }

  Vectorized<T> erf() const {
    return mapSleef(Sleef_erff4_u10, Sleef_erfd2_u10);
  }
  Vectorized<T> erfc() const {
    return mapSleef(Sleef_erfcf4_u15, Sleef_erfcd2_u15);
  }

  Vectorized<T> exp() const {
    return mapSleef(Sleef_expf4_u10, Sleef_expd2_u10);
  }
  Vectorized<T> exp2() const {
    return mapSleef(Sleef_exp2f4_u10, Sleef_exp2d2_u10);
  }
  Vectorized<T> expm1() const {
    return mapSleef(Sleef_expm1f4_u10, Sleef_expm1d2_u10);
  }
  Vectorized<T> exp_u20() const {
    return exp();
  }
  Vectorized<T> fexp_u20() const {
    return exp();
  }

  Vectorized<T> log() const {
    return mapSleef(Sleef_logf4_u10, Sleef_logd2_u10);
  }
  Vectorized<T> log2() const {
    return mapSleef(Sleef_log2f4_u10, Sleef_log2d2_u10);
  }
  Vectorized<T> log10() const {
    return mapSleef(Sleef_log10f4_u10, Sleef_log10d2_u10);
  }
  Vectorized<T> log1p() const {
    return mapSleef(Sleef_log1pf4_u10, Sleef_log1pd2_u10);
  }

  Vectorized<T> sin() const {
    return mapSleef(Sleef_sinf4_u10, Sleef_sind2_u10);
  }
  Vectorized<T> sinh() const {
    return mapSleef(Sleef_sinhf4_u10, Sleef_sinhd2_u10);
  }
  Vectorized<T> cos() const {
    return mapSleef(Sleef_cosf4_u10, Sleef_cosd2_u10);
  }
  Vectorized<T> cosh() const {
    return mapSleef(Sleef_coshf4_u10, Sleef_coshd2_u10);
  }

  Vectorized<T> tan() const {
    return mapSleef(Sleef_tanf4_u10, Sleef_tand2_u10);
  }
  Vectorized<T> tanh() const {
    return mapSleef(Sleef_tanhf4_u10, Sleef_tanhd2_u10);
  }

  Vectorized<T> lgamma() const {
    return mapSleef(Sleef_lgammaf4_u10, Sleef_lgammad2_u10);
  }

  Vectorized<T> atan2(const Vectorized<T>& b) const {
    return mapSleef(Sleef_atan2f4_u10, Sleef_atan2d2_u10, b);
  }
  Vectorized<T> copysign(const Vectorized<T>& sign) const {
    return mapSleef(Sleef_copysignf4, Sleef_copysignd2, sign);
  }
  Vectorized<T> fmod(const Vectorized<T>& q) const {
    return mapSleef(Sleef_fmodf4, Sleef_fmodd2, q);
  }

  Vectorized<T> hypot(const Vectorized<T>& b) const {
    return mapSleef(Sleef_hypotf4_u05, Sleef_hypotd2_u05, b);
  }

  Vectorized<T> pow(const Vectorized<T>& b) const {
    return mapSleef(Sleef_powf4_u10, Sleef_powd2_u10, b);
  }

  Vectorized<T> nextafter(const Vectorized<T>& b) const {
    return mapSleef(Sleef_nextafterf4, Sleef_nextafterd2, b);
  }

  Vectorized<T> erfinv() const {
    return mapOrdinary(calc_erfinv);
  }

  Vectorized<T> digamma() const {
    return mapOrdinary(calc_digamma);
  }

  Vectorized<T> igamma(const Vectorized<T>& x) const {
    return mapOrdinary(calc_igamma, x);
  }

  Vectorized<T> igammac(const Vectorized<T>& x) const {
    return mapOrdinary(calc_igammac, x);
  }

  Vectorized<T> i0() const {
    return mapOrdinary(calc_i0);
  }

  Vectorized<T> i0e() const {
    return mapOrdinary(calc_i0e);
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> minimum(const Vectorized<T>& other) const {
    return {vec_min(_vec0, other._vec0), vec_min(_vec1, other._vec1)};
  }

  /* Propagates NaN if either input is a NaN. */
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> minimum(const Vectorized<T>& other) const {
    Vectorized<T> tmp = {
        vec_min(_vec0, other._vec0), vec_min(_vec1, other._vec1)};
    tmp = blendv(tmp, *this, isnan());
    return blendv(tmp, other, other.isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> maximum(const Vectorized<T>& other) const {
    return {vec_max(_vec0, other._vec0), vec_max(_vec1, other._vec1)};
  }

  /* Propagates NaN if either input is a NaN. */
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> maximum(const Vectorized<T>& other) const {
    Vectorized<T> tmp = {
        vec_max(_vec0, other._vec0), vec_max(_vec1, other._vec1)};
    tmp = blendv(tmp, *this, isnan());
    return blendv(tmp, other, other.isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> clamp_min(const Vectorized<T>& min) const {
    return {vec_max(_vec0, min._vec0), vec_max(_vec1, min._vec1)};
  }

  /* Keeps NaN if actual value is NaN */
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> clamp_min(const Vectorized<T>& min) const {
    Vectorized<T> tmp = {vec_max(_vec0, min._vec0), vec_max(_vec1, min._vec1)};
    return blendv(tmp, *this, isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> clamp_max(const Vectorized<T>& max) const {
    return {vec_min(_vec0, max._vec0), vec_min(_vec1, max._vec1)};
  }

  /* Keeps NaN if actual value is NaN */
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> clamp_max(const Vectorized<T>& max) const {
    Vectorized<T> tmp = {vec_min(_vec0, max._vec0), vec_min(_vec1, max._vec1)};
    return blendv(tmp, *this, isnan());
  }

  template <typename U = T, std::enable_if_t<std::is_same_v<U, float>, int> = 0>
  Vectorized<T> swapped() const {
    auto swap_mask = GetSwapMaskFloat();
    vtype v0 = vec_perm(_vec0, _vec0, swap_mask);
    vtype v1 = vec_perm(_vec1, _vec1, swap_mask);
    return {v0, v1};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same_v<U, double>, int> = 0>
  Vectorized<T> swapped() const {
    vtype v0 = {_vec0[1], _vec0[0]};
    vtype v1 = {_vec1[1], _vec1[0]};
    return {v0, v1};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  static Vectorized<T> mergee(Vectorized<T>& first, Vectorized<T>& second) {
    return {
        vec_mergee(first._vec0, second._vec0),
        vec_mergee(first._vec1, second._vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  static Vectorized<T> mergeo(Vectorized<T>& first, Vectorized<T>& second) {
    return {
        vec_mergeo(first._vec0, second._vec0),
        vec_mergeo(first._vec1, second._vec1)};
  }

  static Vectorized<T> horizontal_add_perm(
      Vectorized<T>& first,
      Vectorized<T>& second) {
    // we will simulate it differently with 6 instructions total
    // lets permute second so that we can add it getting horizontal sums
    auto first_perm = first.swapped(); // 2perm
    auto second_perm = second.swapped(); // 2perm
    // summ
    auto first_ret = first + first_perm; // 2add
    auto second_ret = second + second_perm; // 2 add
    // now lets choose evens
    return mergee(first_ret, second_ret); // 2 mergee's
  }

  static Vectorized<T> horizontal_sub_perm(
      Vectorized<T>& first,
      Vectorized<T>& second) {
    // we will simulate it differently with 6 instructions total
    // lets permute second so that we can add it getting horizontal sums
    auto first_perm = first.swapped(); // 2perm
    auto second_perm = second.swapped(); // 2perm
    // summ
    auto first_ret = first - first_perm; // 2sub
    auto second_ret = second - second_perm; // 2 sub
    // now lets choose evens
    return mergee(first_ret, second_ret); // 2 mergee's
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> mergee() const {
    return {vec_mergee(_vec0, _vec0), vec_mergee(_vec1, _vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
  Vectorized<T> mergeo() const {
    return {vec_mergeo(_vec0, _vec0), vec_mergeo(_vec1, _vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same_v<U, uint8_t>, int> = 0>
  Vectorized<int32_t> to_vec_float_helper() const {
    int32_t values[8] = {
        _vec0[0],
        _vec0[1],
        _vec0[2],
        _vec0[3],
        _vec0[4],
        _vec0[5],
        _vec0[6],
        _vec0[7],
    };

    return Vectorized<int32_t>{
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7]};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same_v<U, int32_t>, int> = 0>
  Vectorized<uint8_t> to_vec_uint8_helper() const {
    // helper function for float to uint8_t conversion
    uint8_t values[8] = {
        static_cast<uint8_t>(_vec0[0]),
        static_cast<uint8_t>(_vec0[1]),
        static_cast<uint8_t>(_vec0[2]),
        static_cast<uint8_t>(_vec0[3]),
        static_cast<uint8_t>(_vec1[0]),
        static_cast<uint8_t>(_vec1[1]),
        static_cast<uint8_t>(_vec1[2]),
        static_cast<uint8_t>(_vec1[3]),
    };

    return Vectorized<uint8_t>{
        values[0], values[1], values[2], values[3], values[4], values[5],
        values[6], values[7], 0,         0,         0,         0,
        0,         0,         0,         0,         0,         0,
        0,         0,         0,         0,         0,         0,
        0,         0,         0,         0,         0,         0,
        0,         0,
    };
  }
};

#define ZVECTOR_OPERATORS(typex)                                        \
  template <>                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator+(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{a.vec0() + b.vec0(), a.vec1() + b.vec1()}; \
  }                                                                     \
                                                                        \
  template <>                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator-(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{a.vec0() - b.vec0(), a.vec1() - b.vec1()}; \
  }                                                                     \
                                                                        \
  template <>                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator*(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{a.vec0() * b.vec0(), a.vec1() * b.vec1()}; \
  }                                                                     \
                                                                        \
  template <>                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator/(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{a.vec0() / b.vec0(), a.vec1() / b.vec1()}; \
  }                                                                     \
                                                                        \
  template <>                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator&(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        (Vectorized<typex>::vtype)(a.vecb0() & b.vecb0()),              \
        (Vectorized<typex>::vtype)(a.vecb1() & b.vecb1())};             \
  }                                                                     \
                                                                        \
  template <>                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator|(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        (Vectorized<typex>::vtype)(a.vecb0() | b.vecb0()),              \
        (Vectorized<typex>::vtype)(a.vecb1() | b.vecb1())};             \
  }                                                                     \
                                                                        \
  template <>                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator^(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        (Vectorized<typex>::vtype)(a.vecb0() ^ b.vecb0()),              \
        (Vectorized<typex>::vtype)(a.vecb1() ^ b.vecb1())};             \
  }                                                                     \
                                                                        \
  Vectorized<typex> C10_ALWAYS_INLINE operator==(                       \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        vec_cmpeq(a.vec0(), b.vec0()), vec_cmpeq(a.vec1(), b.vec1())};  \
  }                                                                     \
                                                                        \
  Vectorized<typex> C10_ALWAYS_INLINE operator!=(                       \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        vec_cmpeq(a.vec0(), b.vec0()), vec_cmpeq(a.vec1(), b.vec1())}   \
        ._not();                                                        \
  }                                                                     \
                                                                        \
  Vectorized<typex> C10_ALWAYS_INLINE operator>(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        vec_cmpgt(a.vec0(), b.vec0()), vec_cmpgt(a.vec1(), b.vec1())};  \
  }                                                                     \
                                                                        \
  Vectorized<typex> C10_ALWAYS_INLINE operator>=(                       \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        vec_cmpge(a.vec0(), b.vec0()), vec_cmpge(a.vec1(), b.vec1())};  \
  }                                                                     \
                                                                        \
  Vectorized<typex> C10_ALWAYS_INLINE operator<(                        \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        vec_cmplt(a.vec0(), b.vec0()), vec_cmplt(a.vec1(), b.vec1())};  \
  }                                                                     \
                                                                        \
  Vectorized<typex> C10_ALWAYS_INLINE operator<=(                       \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {         \
    return Vectorized<typex>{                                           \
        vec_cmple(a.vec0(), b.vec0()), vec_cmple(a.vec1(), b.vec1())};  \
  }

ZVECTOR_OPERATORS(float)
ZVECTOR_OPERATORS(double)
ZVECTOR_OPERATORS(int8_t)
ZVECTOR_OPERATORS(uint8_t)
ZVECTOR_OPERATORS(uint16_t)
ZVECTOR_OPERATORS(int16_t)
ZVECTOR_OPERATORS(int32_t)
ZVECTOR_OPERATORS(int64_t)

#undef ZVECTOR_OPERATORS

#define ZVECTOR_OPERATORS(typex)                                          \
  template <>                                                             \
  Vectorized<typex> C10_ALWAYS_INLINE operator<<(                         \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {           \
    constexpr Vectorized<typex>::ElementType max_shift =                  \
        sizeof(Vectorized<typex>::ElementType) * CHAR_BIT;                \
                                                                          \
    Vectorized<typex>::ElementType a_array[Vectorized<typex>::size()];    \
    Vectorized<typex>::ElementType b_array[Vectorized<typex>::size()];    \
    Vectorized<typex>::ElementType c_array[Vectorized<typex>::size()];    \
                                                                          \
    a.store(a_array);                                                     \
    b.store(b_array);                                                     \
                                                                          \
    for (int i = 0; i != Vectorized<typex>::size(); i++) {                \
      typex shift = b_array[i];                                           \
      if ((static_cast<std::make_signed_t<typex>>(shift) < 0) ||          \
          (shift >= max_shift)) {                                         \
        c_array[i] = 0;                                                   \
      } else {                                                            \
        c_array[i] = static_cast<std::make_unsigned_t<typex>>(a_array[i]) \
            << shift;                                                     \
      }                                                                   \
    }                                                                     \
                                                                          \
    return Vectorized<typex>::loadu(c_array);                             \
  }                                                                       \
                                                                          \
  template <>                                                             \
  Vectorized<typex> C10_ALWAYS_INLINE operator>>(                         \
      const Vectorized<typex>& a, const Vectorized<typex>& b) {           \
    /* right shift value to retain sign bit for signed and no bits for    \
     * unsigned */                                                        \
    constexpr Vectorized<typex>::ElementType max_shift =                  \
        sizeof(typex) * CHAR_BIT - std::is_signed_v<typex>;               \
                                                                          \
    Vectorized<typex>::ElementType a_array[Vectorized<typex>::size()];    \
    Vectorized<typex>::ElementType b_array[Vectorized<typex>::size()];    \
    Vectorized<typex>::ElementType c_array[Vectorized<typex>::size()];    \
                                                                          \
    a.store(a_array);                                                     \
    b.store(b_array);                                                     \
                                                                          \
    for (int i = 0; i != Vectorized<typex>::size(); i++) {                \
      typex shift = b_array[i];                                           \
      if ((static_cast<std::make_signed_t<typex>>(shift) < 0) ||          \
          (shift >= max_shift)) {                                         \
        c_array[i] = a_array[i] >> max_shift;                             \
      } else {                                                            \
        c_array[i] = a_array[i] >> shift;                                 \
      }                                                                   \
    }                                                                     \
                                                                          \
    return Vectorized<typex>::loadu(c_array);                             \
  }                                                                       \
                                                                          \
  template <>                                                             \
  inline Vectorized<typex> operator~(const Vectorized<typex>& a) {        \
    return a._not();                                                      \
  }

ZVECTOR_OPERATORS(int8_t)
ZVECTOR_OPERATORS(uint8_t)
ZVECTOR_OPERATORS(uint16_t)
ZVECTOR_OPERATORS(int16_t)
ZVECTOR_OPERATORS(int32_t)
ZVECTOR_OPERATORS(int64_t)

#undef ZVECTOR_OPERATORS

#define DEFINE_MAXMIN_FUNCS(operand_type)                                     \
  template <>                                                                 \
  Vectorized<operand_type> inline maximum(                                    \
      const Vectorized<operand_type>& a, const Vectorized<operand_type>& b) { \
    return a.maximum(b);                                                      \
  }                                                                           \
  template <>                                                                 \
  Vectorized<operand_type> inline minimum(                                    \
      const Vectorized<operand_type>& a, const Vectorized<operand_type>& b) { \
    return a.minimum(b);                                                      \
  }

#define DEFINE_CLAMP_MAXMIN_FUNCS(typex)                          \
  DEFINE_MAXMIN_FUNCS(typex)                                      \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp_min(                  \
      const Vectorized<typex>& a, const Vectorized<typex>& min) { \
    return a.clamp_min(min);                                      \
  }                                                               \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp_max(                  \
      const Vectorized<typex>& a, const Vectorized<typex>& max) { \
    return a.clamp_max(max);                                      \
  }                                                               \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp(                      \
      const Vectorized<typex>& a,           
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 226 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vec`, `CPU_CAPABILITY`, `at`

**Classes/Structs**: `VecBinaryType`, `VecBinaryType`, `VecBinaryType`, `VecBinaryType`, `VecBinaryType`, `VecInnerType`, `VecInnerType`, `is_vec_specialized_for`, `Vectorized`, `LoaduHelper`, `LoaduHelper`, `StoreHelper`, `StoreHelper`, `unpack_type`, `unpack_type`, `unpack_type`, `unpack_type`, `pack_type`, `pack_type`, `pack_type`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cpu/vec/vec256/zarch`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cmath`
- `cstring`
- `limits`
- `type_traits`
- `utility`
- `sleef.h`
- `sleef.h`
- `vecintrin.h`
- `ATen/cpu/vec/intrinsics.h`
- `ATen/cpu/vec/vec_base.h`
- `c10/util/complex.h`


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

Files in the same folder (`aten/src/ATen/cpu/vec/vec256/zarch`):



## Cross-References

- **File Documentation**: `vec256_zarch.h_docs.md`
- **Keyword Index**: `vec256_zarch.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
