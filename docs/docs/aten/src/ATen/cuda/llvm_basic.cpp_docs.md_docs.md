# Documentation: `docs/aten/src/ATen/cuda/llvm_basic.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/llvm_basic.cpp_docs.md`
- **Size**: 10,426 bytes (10.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/llvm_basic.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cuda/llvm_basic.cpp`
- **Size**: 7,721 bytes (7.54 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// This file is modified from LLVM, see the following copyright information
//
// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>
#include <ATen/cuda/llvm_jit_strings.h>

namespace at::cuda {

// copy-pasted from some llvm files:
// - https://github.com/llvm/llvm-project/blob/main/libcxx/include/type_traits
// - https://github.com/llvm/llvm-project/blob/main/clang/test/Headers/Inputs/include/type_traits
const std::string traits = R"ESCAPE(

namespace std {

template <class _Tp>
_Tp&& __declval(int);
template <class _Tp>
_Tp __declval(long);
template <class _Tp>
decltype(__declval<_Tp>(0)) declval() noexcept;

template <class _Tp, _Tp __v>
struct integral_constant {
  static const _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant type;
};

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

// is_same, functional
template <class _Tp, class _Up> struct is_same : public false_type {};
template <class _Tp> struct is_same<_Tp, _Tp> : public true_type {};

// is_integral, for some types.
template <class _Tp> struct is_integral
    : public integral_constant<bool, false> {};
template <> struct is_integral<bool>
    : public integral_constant<bool, true> {};
template <> struct is_integral<char>
    : public integral_constant<bool, true> {};
template <> struct is_integral<short>
    : public integral_constant<bool, true> {};
template <> struct is_integral<int>
    : public integral_constant<bool, true> {};
template <> struct is_integral<long>
    : public integral_constant<bool, true> {};
template <> struct is_integral<long long>
    : public integral_constant<bool, true> {};

// enable_if, functional
template <bool _C, typename _Tp> struct enable_if{};
template <typename _Tp> struct enable_if<true, _Tp>{
  using type = _Tp;
};
template <bool b, class T=void>
using enable_if_t = typename enable_if<b,T>::type;

template <class _Tp> struct remove_const            {typedef _Tp type;};
template <class _Tp> struct remove_const<const _Tp> {typedef _Tp type;};
template <class _Tp> using remove_const_t = typename remove_const<_Tp>::type;

template <class _Tp> struct remove_volatile               {typedef _Tp type;};
template <class _Tp> struct remove_volatile<volatile _Tp> {typedef _Tp type;};
template <class _Tp> using remove_volatile_t = typename remove_volatile<_Tp>::type;

template <class _Tp> struct remove_cv
{typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;};
template <class _Tp> using remove_cv_t = typename remove_cv<_Tp>::type;

template <class _Tp> struct __libcpp_is_floating_point              : public false_type {};
template <>          struct __libcpp_is_floating_point<float>       : public true_type {};
template <>          struct __libcpp_is_floating_point<double>      : public true_type {};
template <>          struct __libcpp_is_floating_point<long double> : public true_type {};

template <class _Tp> struct is_floating_point
    : public __libcpp_is_floating_point<typename remove_cv<_Tp>::type> {};

template <class _Tp> struct is_arithmetic
    : public integral_constant<bool, is_integral<_Tp>::value      ||
                                     is_floating_point<_Tp>::value> {};
template <class _Tp>
inline constexpr bool is_arithmetic_v = is_arithmetic<_Tp>::value;

template <class _Tp>
struct __numeric_type
{
   static void __test(...);
   static float __test(float);
   static double __test(char);
   static double __test(int);
   static double __test(unsigned);
   static double __test(long);
   static double __test(unsigned long);
   static double __test(long long);
   static double __test(unsigned long long);
   static double __test(double);
   static long double __test(long double);

   typedef decltype(__test(declval<_Tp>())) type;
   static const bool value = !is_same<type, void>::value;
};

template <>
struct __numeric_type<void>
{
   static const bool value = true;
};

// __promote

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&
                 __numeric_type<_A2>::value &&
                 __numeric_type<_A3>::value>
class __promote_imp
{
public:
    static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
    typedef typename __promote_imp<_A3>::type __type3;
public:
    typedef decltype(__type1() + __type2() + __type3()) type;
    static const bool value = true;
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
public:
    typedef decltype(__type1() + __type2()) type;
    static const bool value = true;
};

template <class _A1>
class __promote_imp<_A1, void, void, true>
{
public:
    typedef typename __numeric_type<_A1>::type type;
    static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

} // namespace std

)ESCAPE";

const std::string &get_traits_string() {
    return traits;
}

// This is copy-pasted from the following llvm file:
// - https://github.com/llvm/llvm-project/blob/main/libcxx/include/cmath
const std::string cmath = R"ESCAPE(

namespace std {

using ::signbit;
using ::isfinite;
using ::isinf;
using ::isnan;

using ::abs;

using ::acos;
using ::acosf;
using ::asin;
using ::asinf;
using ::atan;
using ::atanf;
using ::atan2;
using ::atan2f;
using ::ceil;
using ::ceilf;
using ::cos;
using ::cosf;
using ::cosh;
using ::coshf;

using ::exp;
using ::expf;

using ::fabs;
using ::fabsf;
using ::floor;
using ::floorf;

using ::fmod;
using ::fmodf;

using ::frexp;
using ::frexpf;
using ::ldexp;
using ::ldexpf;

using ::log;
using ::logf;

using ::log10;
using ::log10f;
using ::modf;
using ::modff;

using ::pow;
using ::powf;

using ::sin;
using ::sinf;
using ::sinh;
using ::sinhf;

using ::sqrt;
using ::sqrtf;
using ::tan;
using ::tanf;

using ::tanh;
using ::tanhf;

using ::acosh;
using ::acoshf;
using ::asinh;
using ::asinhf;
using ::atanh;
using ::atanhf;
using ::cbrt;
using ::cbrtf;

using ::copysign;
using ::copysignf;

using ::erf;
using ::erff;
using ::erfc;
using ::erfcf;
using ::exp2;
using ::exp2f;
using ::expm1;
using ::expm1f;
using ::fdim;
using ::fdimf;
using ::fmaf;
using ::fma;
using ::fmax;
using ::fmaxf;
using ::fmin;
using ::fminf;
using ::hypot;
using ::hypotf;
using ::ilogb;
using ::ilogbf;
using ::lgamma;
using ::lgammaf;
using ::llrint;
using ::llrintf;
using ::llround;
using ::llroundf;
using ::log1p;
using ::log1pf;
using ::log2;
using ::log2f;
using ::logb;
using ::logbf;
using ::lrint;
using ::lrintf;
using ::lround;
using ::lroundf;

using ::nan;
using ::nanf;

using ::nearbyint;
using ::nearbyintf;
using ::nextafter;
using ::nextafterf;
using ::remainder;
using ::remainderf;
using ::remquo;
using ::remquof;
using ::rint;
using ::rintf;
using ::round;
using ::roundf;
using ::scalbln;
using ::scalblnf;
using ::scalbn;
using ::scalbnf;
using ::tgamma;
using ::tgammaf;
using ::trunc;
using ::truncf;

} // namespace std

)ESCAPE";

const std::string &get_cmath_string() {
    return cmath;
}

} // namespace at::cuda

```



## High-Level Overview


This C++ file contains approximately 39 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `std`, `at`

**Classes/Structs**: `_Tp`, `_Tp`, `_Tp`, `_Tp`, `integral_constant`, `_Tp`, `_Up`, `is_same`, `_Tp`, `is_same`, `_Tp`, `is_integral`, `is_integral`, `is_integral`, `is_integral`, `is_integral`, `is_integral`, `is_integral`, `enable_if`, `enable_if`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `string`
- `ATen/cuda/llvm_jit_strings.h`


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

Files in the same folder (`aten/src/ATen/cuda`):

- [`CublasHandlePool.cpp_docs.md`](./CublasHandlePool.cpp_docs.md)
- [`CUDABlas.h_docs.md`](./CUDABlas.h_docs.md)
- [`jiterator.cu_docs.md`](./jiterator.cu_docs.md)
- [`CUDAGraph.h_docs.md`](./CUDAGraph.h_docs.md)
- [`llvm_jit_strings.h_docs.md`](./llvm_jit_strings.h_docs.md)
- [`llvm_complex.cpp_docs.md`](./llvm_complex.cpp_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md`](./CUDAGeneratorImpl.cpp_docs.md)
- [`cub_definitions.cuh_docs.md`](./cub_definitions.cuh_docs.md)
- [`jiterator_impl.h_docs.md`](./jiterator_impl.h_docs.md)


## Cross-References

- **File Documentation**: `llvm_basic.cpp_docs.md`
- **Keyword Index**: `llvm_basic.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/aten/src/ATen/cuda`):

- [`PhiloxCudaState.h_docs.md_docs.md`](./PhiloxCudaState.h_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md_docs.md`](./CUDAGeneratorImpl.cpp_docs.md_docs.md)
- [`Exceptions.cpp_docs.md_docs.md`](./Exceptions.cpp_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_kw.md_docs.md`](./CUDAGeneratorImpl.cpp_kw.md_docs.md)
- [`Sleep.h_docs.md_docs.md`](./Sleep.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-2.cu_kw.md_docs.md`](./cub-RadixSortPairs-int64-2.cu_kw.md_docs.md)
- [`CUDASparseDescriptors.h_kw.md_docs.md`](./CUDASparseDescriptors.h_kw.md_docs.md)
- [`jiterator_impl.h_docs.md_docs.md`](./jiterator_impl.h_docs.md_docs.md)
- [`CUDAContext.h_docs.md_docs.md`](./CUDAContext.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-4.cu_docs.md_docs.md`](./cub-RadixSortPairs-int64-4.cu_docs.md_docs.md)


## Cross-References

- **File Documentation**: `llvm_basic.cpp_docs.md_docs.md`
- **Keyword Index**: `llvm_basic.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
