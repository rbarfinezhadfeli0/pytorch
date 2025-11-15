# Documentation: `docs/aten/src/ATen/cpu/vec/vec_mask.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cpu/vec/vec_mask.h_docs.md`
- **Size**: 12,357 bytes (12.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cpu/vec/vec_mask.h`

## File Metadata

- **Path**: `aten/src/ATen/cpu/vec/vec_mask.h`
- **Size**: 9,971 bytes (9.74 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_n.h>
namespace at::vec {
inline namespace CPU_CAPABILITY {

/**
 * The `VecMask` class provides a convenient interface for working with
 * vectorized masks in SIMD operations. It encapsulates a `Vectorized<T, N>`
 * mask that can be directly usable in masked vectorized operations. It provides
 * various methods for manipulating and accessing the mask elements:
 * 1. `from` and `to`: Conversion between a vector of boolean values and a
 * vectorized mask.
 * 2. `cast`: Casts the mask to a different base type.
 * 3. `all_zero`: Checks if all mask elements are zero.
 * 4. `is_masked`: Checks if a specific element is masked.
 * 5. `loadu`: Loads data from memory using the mask.
 * 6. `all_masked`: Checks if all mask elements are masked.
 *
 * Some helper template classes are provided to simplify the specialization of
 * the `VecMask` for the specific CPU arch:
 * 1. `VecMaskLoad`: Loads data from memory using the mask.
 * 2. `VecMaskTo`: Converts the mask to boolean.
 * 3. `VecMaskCast`: Casts the mask to a different base type.
 *
 */
template <typename T, int N>
class VecMask;

template <
    typename data_t,
    int data_n,
    typename mask_t,
    int mask_n,
    typename Enabled = void>
struct VecMaskLoad {
  static inline VectorizedN<data_t, data_n> apply(
      const data_t* ptr,
      const VecMask<mask_t, mask_n>& vec_mask) {
    constexpr typename VecMask<mask_t, mask_n>::size_type size =
        VecMask<mask_t, mask_n>::size();
    static_assert(VectorizedN<data_t, data_n>::size() >= size);
    __at_align__ data_t data[size];
    __at_align__ mask_t mask[size];
    auto mask_ = VectorizedN<mask_t, mask_n>(vec_mask);
    mask_.store(mask);
    for (int i = 0; i < size; i++) {
      data[i] = mask[i] ? ptr[i] : static_cast<data_t>(0);
    }
    return VectorizedN<data_t, data_n>::loadu(data, size);
  }
};

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    typename Enabled = void>
struct VecMaskTo {
  static inline VecMask<dst_t, dst_n> apply(
      const VecMask<src_t, src_n>& vec_mask) {
    auto zeros = VectorizedN<dst_t, dst_n>(static_cast<dst_t>(0));
    auto ones = VectorizedN<dst_t, dst_n>(static_cast<dst_t>(1));
    return VectorizedN<dst_t, dst_n>::blendv(
        zeros, ones, vec_mask.template cast<dst_t, dst_n>());
  }
};

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    typename Enabled = void>
struct VecMaskCast {
  static inline VecMask<dst_t, dst_n> apply(
      const VecMask<src_t, src_n>& vec_mask) {
    return VecMask<dst_t, dst_n>::from(VectorizedN<src_t, src_n>(vec_mask));
  }
};

template <typename T, int N>
struct VecMaskCast<T, N, T, N> {
  static inline VecMask<T, N> apply(const VecMask<T, N>& vec_mask) {
    return vec_mask;
  }
};

template <typename T, int N>
struct VecMaskCheck {
  static inline bool all_zero(const VectorizedN<T, N>& vec_mask) {
    __at_align__ T mask[VectorizedN<T, N>::size()];
    vec_mask.store(mask);
    return std::all_of(mask, mask + VectorizedN<T, N>::size(), [](T m) {
      return m == static_cast<T>(0);
    });
  }

  static inline bool all_masked(const VectorizedN<T, N>& vec_mask) {
    __at_align__ T mask[VectorizedN<T, N>::size()];
    vec_mask.store(mask);
    return std::all_of(mask, mask + VectorizedN<T, N>::size(), [](T m) {
      return m != static_cast<T>(0);
    });
  }

  static inline bool is_masked(const VectorizedN<T, N>& vec_mask, int i) {
    __at_align__ T mask[VectorizedN<T, N>::size()];
    vec_mask.store(mask);
    return mask[i] != static_cast<T>(0);
  }
};

template <typename T, int N>
class VecMask {
 public:
  using size_type = int;
  static constexpr size_type size() {
    return VectorizedN<T, N>::size();
  }

 private:
  VectorizedN<T, N> mask_;

 public:
  VecMask() : mask_(static_cast<T>(0)) {}
  VecMask(const VectorizedN<T, N>& mask) : mask_(mask) {}

  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  VecMask(const Vectorized<T>& mask) : mask_(mask) {}

  template <typename U, int L>
  static VecMask<T, N> from(const VectorizedN<U, L>& b_vec) {
    __at_align__ U b_buf[size()];
    if constexpr (size() >= VectorizedN<U, L>::size()) {
      b_vec.store(b_buf);
      for (int i = VectorizedN<U, L>::size(); i < size(); i++) {
        b_buf[i] = static_cast<U>(0);
      }
    } else {
      b_vec.store(b_buf, size());
    }
    return from(b_buf);
  }

  template <typename U>
  static VecMask<T, N> from(U b) {
    using int_t = int_same_size_t<T>;
    T mask = b ? c10::bit_cast<T>((int_t)(~(int_t)0)) : (T)0;
    return VectorizedN<T, N>(mask);
  }

  template <typename U>
  static VecMask<T, N> from(U* b) {
    using int_t = int_same_size_t<T>;
    __at_align__ T mask[size()];
#ifndef __msvc_cl__
#pragma unroll
#endif
    for (int i = 0; i < size(); i++) {
      *(int_t*)(mask + i) = b[i] ? ~(int_t)0 : (int_t)0;
    }
    return VectorizedN<T, N>(VectorizedN<T, N>::loadu(mask));
  }

  static VecMask<T, N> blendv(
      const VecMask<T, N>& c,
      const VecMask<T, N>& b,
      const VecMask<T, N>& a) {
    VectorizedN<T, N> result = VectorizedN<T, N>::blendv(
        VectorizedN<T, N>(c), VectorizedN<T, N>(b), VectorizedN<T, N>(a));
    return result;
  }

  static VecMask<T, N> set(
      const VecMask<T, N>& a,
      const VecMask<T, N>& b,
      int64_t count = size()) {
    VectorizedN<T, N> result = VectorizedN<T, N>::set(
        VectorizedN<T, N>(a), VectorizedN<T, N>(b), count);
    return result;
  }

  void store(bool* b, int count = size()) {
    constexpr int L =
        (VectorizedN<T, N>::size() + Vectorized<bool>::size() - 1) /
        Vectorized<bool>::size();
    auto res = this->to<bool, L>();
    res.store(b, count);
    return;
  }

  template <typename U, int L, std::enable_if_t<L >= 2, int> = 0>
  inline VectorizedN<U, L> to() const {
    return VecMaskTo<U, L, T, N>::apply(*this);
  }

  template <typename U, int L, std::enable_if_t<L == 1, int> = 0>
  inline Vectorized<U> to() const {
    return VecMaskTo<U, L, T, N>::apply(*this);
  }

  template <typename U, int L>
  inline VecMask<U, L> cast() const {
    return VecMaskCast<U, L, T, N>::apply(*this);
  }

  inline bool all_zero() const {
    return VecMaskCheck<T, N>::all_zero(mask_);
  }

  inline bool all_masked() const {
    return VecMaskCheck<T, N>::all_masked(mask_);
  }

  inline bool is_masked(int i) const {
    return VecMaskCheck<T, N>::is_masked(mask_, i);
  }

  inline operator VectorizedN<T, N>() const {
    return mask_;
  }

  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  inline operator Vectorized<T>() const {
    return mask_[0];
  }

  inline Vectorized<T> operator[](int i) const {
    return mask_[i];
  }

  template <
      typename U,
      int L,
      std::enable_if_t<L >= 2 && VectorizedN<U, L>::size() >= size(), int> = 0>
  VectorizedN<U, L> loadu(const U* ptr) const {
    return VecMaskLoad<U, L, T, N>::apply(ptr, *this);
  }

  template <
      typename U,
      int L,
      std::enable_if_t<L == 1 && Vectorized<U>::size() >= size(), int> = 0>
  Vectorized<U> loadu(const U* ptr) const {
    return VecMaskLoad<U, L, T, N>::apply(ptr, *this);
  }
};

#define VEC_MASK_DEFINE_UNARY_OP_GLOBAL(op)         \
  template <typename T, int N>                      \
  inline VecMask<T, N> op(const VecMask<T, N>& a) { \
    return op(VectorizedN<T, N>(a));                \
  }

#define VEC_MASK_DEFINE_BINARY_OP_GLOBAL(op)                                  \
  template <                                                                  \
      typename T,                                                             \
      int N,                                                                  \
      typename V,                                                             \
      int M,                                                                  \
      std::enable_if_t<VecMask<T, N>::size() == VecMask<V, M>::size(), int> = \
          0>                                                                  \
  inline VecMask<T, N> op(const VecMask<T, N>& a, const VecMask<V, M>& b) {   \
    return op(                                                                \
        VectorizedN<T, N>(a), VectorizedN<T, N>(b.template cast<T, N>()));    \
  }

#define VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(op, EXPR)                  \
  template <                                                                  \
      typename T,                                                             \
      int N,                                                                  \
      typename V,                                                             \
      int M,                                                                  \
      std::enable_if_t<VecMask<T, N>::size() == VecMask<V, M>::size(), int> = \
          0>                                                                  \
  inline VecMask<T, N> op(const VecMask<T, N>& a, const VecMask<V, M>& b) {   \
    return EXPR;                                                              \
  }

VEC_MASK_DEFINE_UNARY_OP_GLOBAL(operator~)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator&)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator|)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator^)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator*)
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator>, a & ~b)
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator<, ~a& b)
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator==, ~(a ^ b))
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator>=, (a == b) | (a > b))
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator<=, (a == b) | (a < b))
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator!=, (a ^ b))

#undef VEC_MASK_DEFINE_UNARY_OP_GLOBAL
#undef VEC_MASK_DEFINE_BINARY_OP_GLOBAL
#undef VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL

} // namespace CPU_CAPABILITY
} // namespace at::vec

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `CPU_CAPABILITY`, `at`

**Classes/Structs**: `provides`, `VecMask`, `VecMaskLoad`, `VecMaskTo`, `VecMaskCast`, `VecMaskCast`, `VecMaskCheck`, `VecMask`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cpu/vec`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cpu/vec/vec_base.h`
- `ATen/cpu/vec/vec_n.h`


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

Files in the same folder (`aten/src/ATen/cpu/vec`):

- [`functional_bfloat16.h_docs.md`](./functional_bfloat16.h_docs.md)
- [`vec_n.h_docs.md`](./vec_n.h_docs.md)
- [`functional_base.h_docs.md`](./functional_base.h_docs.md)
- [`intrinsics.h_docs.md`](./intrinsics.h_docs.md)
- [`vec_quant.h_docs.md`](./vec_quant.h_docs.md)
- [`vec_base.h_docs.md`](./vec_base.h_docs.md)
- [`functional.h_docs.md`](./functional.h_docs.md)
- [`vec_half.h_docs.md`](./vec_half.h_docs.md)
- [`vec.h_docs.md`](./vec.h_docs.md)


## Cross-References

- **File Documentation**: `vec_mask.h_docs.md`
- **Keyword Index**: `vec_mask.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cpu/vec`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cpu/vec`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/cpu/vec`):

- [`vec_half.h_kw.md_docs.md`](./vec_half.h_kw.md_docs.md)
- [`functional_bfloat16.h_docs.md_docs.md`](./functional_bfloat16.h_docs.md_docs.md)
- [`functional.h_docs.md_docs.md`](./functional.h_docs.md_docs.md)
- [`vec_mask.h_kw.md_docs.md`](./vec_mask.h_kw.md_docs.md)
- [`vec_n.h_kw.md_docs.md`](./vec_n.h_kw.md_docs.md)
- [`vec_base.h_kw.md_docs.md`](./vec_base.h_kw.md_docs.md)
- [`vec.h_kw.md_docs.md`](./vec.h_kw.md_docs.md)
- [`intrinsics.h_docs.md_docs.md`](./intrinsics.h_docs.md_docs.md)
- [`functional_base.h_docs.md_docs.md`](./functional_base.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `vec_mask.h_docs.md_docs.md`
- **Keyword Index**: `vec_mask.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
